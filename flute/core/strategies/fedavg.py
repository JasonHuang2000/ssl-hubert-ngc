# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os

import torch

from utils import compute_grad_cosines, print_rank
from core.strategies import BaseStrategy
from core.strategies.utils import (
    aggregate_gradients_inplace,
)

from azureml.core import Run
run = Run.get_context()


class FedAvg(BaseStrategy):
    '''Federated Averaging'''

    def __init__(self, mode, config, model_path=None):
        '''Federated Averaging strategy.

        Args:
            mode (str): which part the instantiated object should play,
                typically either :code:`client` or :code:`server`.
            config (dict): initial config dict.
            model_path (str): where to find model, needed for debugging only.
        '''

        super().__init__(mode=mode, config=config, model_path=model_path)

        if mode not in ['client', 'server']:
            raise ValueError('mode in strategy must be either `client` or `server`')

        self.config = config
        self.model_path = model_path
        self.mode = mode

        # Parse config
        self.model_config = config['model_config']
        self.client_config = config['client_config']
        self.server_config = config['server_config']

        self.dp_config = config.get('dp_config', None)

        if mode == 'client':
            self.stats_on_smooth_grad = self.client_config.get('stats_on_smooth_grad', False)
        elif mode == 'server':
            self.dump_norm_stats = self.config.get('dump_norm_stats', False)
            self.aggregate_fast = self.server_config.get('fast_aggregation', False)

            self.skip_model_update = False

            # Initialize accumulators
            self.client_parameters_stack = []
            self.client_weights = []

    def generate_client_payload(self, trainer):
        '''Generate client payload

        Args:
            trainer (core.Trainer object): trainer on client.

        Returns:
            dict containing payloads in some specified format.
        '''

        if self.mode != 'client':
            raise RuntimeError('this method can only be invoked by the client')

        # Reset gradient stats and recalculate them on the smooth/pseudo gradient
        if self.stats_on_smooth_grad:
            trainer.reset_gradient_power()
            trainer.estimate_sufficient_stats()

        # Weight the gradient and remove gradients of the layers we want to freeze
        weight = trainer.num_samples # number of samples per epoch
        for n, p in trainer.model.named_parameters():
            p.grad = weight * p.grad
            if self.model_config.get('freeze_layer', None) and n == self.model_config['freeze_layer']:
                print_rank('Setting gradient to zero for layer: {}'.format(n), loglevel=logging.INFO)
                p.grad.mul_(0)

        payload = {}
        payload['weight'] = weight
        payload['gradients'] = [p.grad.to(torch.device('cpu')) for p in trainer.model.parameters()]

        return payload

    def process_individual_payload(self, worker_trainer, payload):
        '''Process client payload

        Args:
            worker_trainer (core.Trainer object): trainer on server
                (aka model updater).
            payload (dict): whatever is generated by
                :code:`generate_client_payload`.

        Returns:
            True if processed succesfully, False otherwise.
        '''

        if self.mode != 'server':
            raise RuntimeError('this method can only be invoked by the server')

        if payload['weight'] == 0.0:
            return False

        self.client_weights.append(payload['weight'])
        if self.aggregate_fast:
            aggregate_gradients_inplace(worker_trainer.model, payload['gradients'])
        else:
            self.client_parameters_stack.append(payload['gradients'])
        return True

    def combine_payloads(self, worker_trainer, curr_iter, num_clients_curr_iter, total_clients, client_stats, logger=None):
        '''Combine payloads to update model

        Args:
            worker_trainer (core.Trainer object): trainer on server
                (aka model updater).
            curr_iter (int): current iteration.
            num_clients_curr_iter (int): number of clients on current iteration.
            client_stats (dict): stats being collected.
            logger (callback): function called to log quantities.

        Returns:
            losses, computed for use with LR scheduler.
        '''

        if self.mode != 'server':
            raise RuntimeError('this method can only be invoked by the server')

        # Aggregation step
        if self.dump_norm_stats:
            cps_copy = [[g.clone().detach() for g in x] for x in self.client_parameters_stack]
        weight_sum = self._aggregate_gradients(worker_trainer, num_clients_curr_iter, self.client_weights, metric_logger=logger)
        print_rank('Sum of weights: {}'.format(weight_sum), loglevel=logging.DEBUG)

        torch.cuda.empty_cache()

        # Normalize with weight_sum
        for p in worker_trainer.model.parameters():
            p.grad /= weight_sum

        if self.dump_norm_stats:
            cosines = compute_grad_cosines(cps_copy, [p.grad.clone().detach() for p in worker_trainer.model.parameters()])
            with open(os.path.join(self.model_path, 'cosines.txt'), 'a', encoding='utf-8') as outfile:
                outfile.write('{}\n'.format(json.dumps(cosines)))

        if self.skip_model_update is True:
            print_rank('Skipping model update')
            return

        # Run optimization with gradient/model aggregated from clients
        print_rank('Updating model')
        worker_trainer.update_model()
        print_rank('Updating learning rate scheduler')
        losses = worker_trainer.run_lr_scheduler(force_run_val=False)

        # TODO: Global DP. See dga.py

        return losses

    def _aggregate_gradients(self, worker_trainer, num_clients_curr_iter, client_weights, metric_logger=None):
        '''Go through stored gradients, aggregate and put them inside model.

        Args:
            num_clients_curr_iter (int): how many clients were processed.
            client_weights: weight for each client.
            metric_logger (callback, optional): callback used for logging.
                Defaults to None, in which case AML logger is used.

        Returns:
            float: sum of weights for all clients.
        '''

        if metric_logger is None:
            metric_logger = run.log

        if not self.aggregate_fast:
            for client_parameters in self.client_parameters_stack:
                # Model parameters are already multiplied with weight on client, we only have to sum them up
                aggregate_gradients_inplace(worker_trainer.model, client_parameters)
        weight_sum = sum(client_weights)

        # Some cleaning
        self.client_parameters_stack = []
        self.client_weights = []

        return weight_sum