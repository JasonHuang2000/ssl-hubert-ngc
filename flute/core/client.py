# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
The Client object is short-lived, instantiated inside workers 1 to N for 
processing a given client's data. It's main method is the `process_round` 
function, used to update the model given a client's data.
'''

import copy
import logging
import os
import time

from easydict import EasyDict as edict
from importlib.machinery import SourceFileLoader
import numpy as np
import torch

# Internal imports
import core.federated as federated
from .strategies import select_strategy
from .trainer import (
    Trainer,
    run_validation_generic,
    set_component_wise_lr,
)
from utils import (
    ScheduledSamplingScheduler,
    make_optimizer,
    print_rank,
    to_device,
    convex_inference,
    alpha_update,
)
from utils.dataloaders_utils import (
    make_train_dataloader,
    make_val_dataloader,
    make_test_dataloader,
    get_dataset,
)
import extensions.privacy
from extensions.privacy import metrics as privacy_metrics
from experiments import make_model

from fairseq import checkpoint_utils
from fairseq.file_io import PathManager
from .fairseq_trainer import Trainer as fairseq_trainer
from .fairseq_train import fairseq_train
from fairseq.logging import meters, metrics, progress_bar

global train_dataset
global trainset_unlab
global trainset_unlab_rand

class Client:
    # It's unclear why, but sphinx refuses to generate method docs
    # if there is no docstring for this class.
    """Client class for specifying individual client training tasks"""

    def __init__(self, client_id, config, send_gradients):
        '''
        Client side processing: computing gradients, update the model and send them back to the server

        Args:
            client_id (int): identifier for grabbing that client's data.
            config (dict): dictionary with parameters loaded from config file.
            send_gradients (bool): if True, model gradients are sent back;
                otherwise, model weights are sent back.
        '''
        super().__init__()
        
        self.client_id = client_id
        self.config = copy.deepcopy(config)
        self.send_gradients = send_gradients

    def get_client_data(self, dataset=None):
        '''"Getter" method that returns all object's attributes at once.'''

        # client_data = self.get_data(self.client_id, dataset)
        return self.client_id, self.config, self.send_gradients

    @staticmethod
    def get_train_dataset(data_path, client_train_config, task):
        '''This function will obtain the dataset for all training
        users.

        Args:
            data_path (str): path to file containing taining data.
            client_train_config (dict): trainig data config.
            task (str): task name.
        '''
        global train_dataset
        global trainset_unlab
        global trainset_unlab_rand

        train_dataset = get_dataset(data_path, client_train_config, task, mode="train")

        if task == 'semisupervision':
            trainset_unlab = get_dataset(data_path, client_train_config, task, mode="train", user_idx = -2)
            trainset_unlab_rand = get_dataset(data_path, client_train_config, task, mode="train", user_idx = -3)
        else:
            trainset_unlab = None
            trainset_unlab_rand = None

        return len(train_dataset.user_list)

    @staticmethod
    def get_data(clients, dataset):
        ''' Create training dictionary'''

        if dataset == None: # Training case
            datasets = [train_dataset, trainset_unlab, trainset_unlab_rand] if trainset_unlab != None else [train_dataset]
        else: # Evaluation case
            datasets = [dataset]

        data_with_labels = hasattr(datasets[0],"user_data_label")
        
        strcts = [] # Returning list length will always be 1 except when the task is semisupervision
        for dataset in datasets:
            input_strct = {'users': [], 'num_samples': [],'user_data': dict(), 'user_data_label': dict()} if data_with_labels else {'users': [], 'num_samples': [],'user_data': dict()}
            for client in clients:
                user = dataset.user_list[client]
                input_strct['users'].append(user)
                input_strct['num_samples'].append(dataset.num_samples[client])
                input_strct['user_data'][user]= dataset.user_data[user]
                if data_with_labels: 
                    input_strct['user_data_label'][user] = dataset.user_data_label[user]
            strcts.append(edict(input_strct))
        
        return strcts 

    @staticmethod
    def run_testvalidate(client_data, server_data, mode, model):
        '''Called by worker to run test/validation sample on a client.

        This functions assumes set_model_for_round has already been called to
        push the model to the client (see federated.py).

        Args:
            client_data (tuple): client data and config. It is a tuple with 3
                components; importantly, the second component is a dict
                containing the data, and the third component is a dict with the
                config parsed from the YAML file.
            server_data (tuple): server data (model parameters mostly). It is
                a tuple with 2 components; importantly, the second component
                consists of the current model parameters.
            mode (str): whether to `test` or `validate`.
            model (torch.nn.Module): actual model without parameters.
        '''

        # Process inputs and initialize variables
        _, data_strcts, config, _ = client_data
        _, model_parameters, iteration = server_data
        config = copy.deepcopy(config)
        model_path = config["model_path"]

        begin = time.time()  

        # Use the server's data config since we're distributing test/validate from the server
        data_strct = data_strcts[0]
        data_config = config['server_config']['data_config'][mode]
        want_logits = data_config.get('wantLogits', False)
        send_dicts = config['server_config'].get('send_dicts', False)

        # Create dataloader 
        dataloader = None
        print_rank('making dataloader with task {}'.format(config['server_config']['task']), loglevel=logging.DEBUG)
        if mode == 'test':
            dataloader = make_test_dataloader(data_config, data_path=None, task=config['server_config']['task'], data_strct=data_strct)
        elif mode == 'val':
            dataloader = make_val_dataloader(data_config, data_path=None, task=config['server_config']['task'], data_strct=data_strct)

        # Set model parameters
        n_layers, n_params = len([f for f in model.parameters()]), len(model_parameters)
        print_rank(f'Copying model parameters... {n_layers}/{n_params}', loglevel=logging.DEBUG)
        
        model = to_device(model)
        
        if send_dicts: # Send model state dictionary
            tmp = {}
            for param_key, param_dict in zip (model.state_dict(), model_parameters):
                tmp[param_key] = param_dict
            model.load_state_dict(tmp)
        else: # Send parameters
            for p, data in zip(model.parameters(), model_parameters):
                p.data = data.detach().clone().cuda() if torch.cuda.is_available() else data.detach().clone()

        print_rank(f'Model setup complete. {time.time() - begin}s elapsed.', loglevel=logging.DEBUG)

        # Compute output and metrics on the test or validation data
        num_instances = sum(data_strct['num_samples'])
        print_rank(f'Validating {num_instances}', loglevel=logging.DEBUG)
        output, metrics = run_validation_generic(model, dataloader)
        
        # Load local model if necessary
        if config['server_config']['type']=='personalization':

            local_model = make_model(config['model_config'])
            user = data_strct['users'][0]

            local_model_name = os.path.join(model_path, user + '_model.tar')

            if os.path.exists(local_model_name):
                print_rank('Loading Local Model .. {}'.format(local_model_name))
                checkpoint = torch.load(local_model_name)
                local_model.load_state_dict(checkpoint["model_state_dict"])

                local_alpha_name = os.path.join(model_path, user + '_alpha')
                if os.path.exists(local_alpha_name):
                    alpha = torch.load(local_alpha_name)
                    print_rank('Loading Alpha Weight from {}: Value={}'.format(local_model_name, alpha))

                    # Run inference and get logits back
                    if mode == 'test':
                        dataloader = make_test_dataloader(data_config, data_path=None, task=config['server_config']['task'], data_strct=data_strct)
                    elif mode == 'val':
                        dataloader = make_val_dataloader(data_config, data_path=None, task=config['server_config']['task'], data_strct=data_strct)

                    output_local, local_metrics = run_validation_generic(local_model, dataloader)
                    loss_local = local_metrics['loss']['value']
                    cer = local_metrics['acc']['value']
                    # Combine logits
                    cer =convex_inference(output, output_local, alpha=alpha)
                    metrics['loss']['value'] = (metrics['loss']['value'] + loss_local) / 2 
                    metrics['acc']['value'] = cer
        output = None if not want_logits else output

        return output, metrics, num_instances



    @staticmethod
    def process_round(fairseq_data, client_data, server_data, model, data_path, eps=1e-7):
        '''Compute gradients given client's data and update model.

        Args:
            client_data (tuple): client data and config. It is a tuple
                consisting of 4 components: an int indicating the client's id, a
                dict containing that client's data, a dict with the config
                parsed from the YAML file, and a bool indicating whether or not
                gradients should be sent.
            server_data (tuple): server data (model parameters mostly). It is
                a tuple consisting of 2 components; importantly, the first is
                a float giving the client's learning rate, and the second a list
                of torch.Tensor's with current model parameters. 
            model (torch.nn.Module): actual model without parameters.
            data_path (str): where to get data from.
            eps (float): lower bound for aggregation weights.
        '''

        # Ensure the client is assigned to the correct GPU
        if torch.cuda.is_available() and torch.cuda.device_count() == federated.size():
            torch.cuda.set_device(federated.local_rank())

        # Process inputs and initialize variables
        fairseq_cfg, task, criterion, quantizer = fairseq_data
        client_id, config, send_gradients = client_data
        initial_lr, model_parameters, iteration = server_data
        config = copy.deepcopy(config)


        model_config = config['model_config']
        client_config = config['client_config']
        data_config = client_config['data_config']['train']
        semisupervision_config = client_config.get('semisupervision',None)
        # task = client_config.get('task', {})
        trainer_config = client_config.get('trainer_config', {})
        privacy_metrics_config = config.get('privacy_metrics_config', None)
        model_path = config["model_path"]

        StrategyClass = select_strategy(config['strategy'])
        strategy = StrategyClass('client', config)
        print_rank(f'Client successfully instantiated strategy {strategy}', loglevel=logging.DEBUG)
        send_dicts = config['server_config'].get('send_dicts', False)

        begin = time.time()  
        client_stats = {}  

        # data_strct = data_strcts[0]
        # user = data_strct['users'][0]
        print_rank('Loading : {}-th client, {}s elapsed'.format(
            client_id[0], time.time() - begin), loglevel=logging.INFO)

        

        # Get dataloaders
        # train_dataloader = make_train_dataloader(data_config, data_path, task=task, clientx=0, data_strct=data_strct)

        # # Instantiate the model object
        # if model is None:
        #     model = make_model(
        #         model_config,
        #         dataloader_type=train_dataloader.__class__.__name__,
        #         input_dim=data_config['input_dim'],
        #         vocab_size=train_dataloader.vocab_size,
        #     )

        # Set model parameters
        n_layers, n_params = len([f for f in model.parameters()]), len(model_parameters)
        print_rank(f'Copying model parameters... {n_layers}/{n_params}', loglevel=logging.DEBUG)
        model = to_device(model)
        

        if send_dicts: # Send model state dictionary
            print_rank(f'Loading model parameters from dictionary...')
            tmp = {}
            for param_key, param_dict in zip (model.state_dict(), model_parameters):
                tmp[param_key] = param_dict
            model.load_state_dict(tmp)
        else: # Send parameters
            for p, data in zip(model.parameters(), model_parameters):
                p.data = data.detach().clone().cuda() if torch.cuda.is_available() else data.detach().clone()
        print_rank(f'Model setup complete. {time.time() - begin}s elapsed.', loglevel=logging.DEBUG)

        # initialize the fairseq trainer
        fs_trainer = fairseq_trainer(fairseq_cfg, task, model, criterion, quantizer)
            

        # initialize the model updater
        update_model = task.build_model(fairseq_cfg.model) 
        update_optimizer = torch.optim.Adam(update_model.parameters(), 
                                            lr=0.0005, 
                                            betas=(0.9,0.98), 
                                            eps=1e-06, 
                                            weight_decay=0.01, 
                                            amsgrad=False)
        update_model.cuda()
        update_model.train()
        update_model.zero_grad()


        # # Make the trainer
        # trainer = Trainer(
        #     model=model,
        #     optimizer=None,
        #     ss_scheduler=None,
        #     train_dataloader=None,
        #     server_replay_config =client_config,
        #     max_grad_norm=client_config['data_config']['train'].get('max_grad_norm', None),
        #     anneal_config=client_config['annealing_config'] if 'annealing_config' in client_config else None,
        #     num_skips_threshold=client_config['num_skips_threshold'] if 'num_skips_threshold' in client_config else -1,
        #     ignore_subtask=client_config['ignore_subtask']
        # )

        print_rank(f'Fairseq setup complete', loglevel=logging.INFO)

        # # Fix parameters of layers
        # if 'updatable_names' in trainer_config:
        #     set_component_wise_lr(model, client_config['optimizer_config'], trainer_config['updatable_names'])

        # Create the optimizer on the workers
        # NOTE: the server dictates the learning rate for the clients
        # client_config['optimizer_config']['lr'] = initial_lr
        # optimizer = make_optimizer(client_config['optimizer_config'], model)

        # Make the scheduled sampling scheduler
        # ss_scheduler = None
        # if 'ss_config' in client_config and client_config['ss_config'] is not None:
        #     ss_scheduler = ScheduledSamplingScheduler(model=model, **client_config['ss_config'])

        # # Make the trainer
        # trainer = Trainer(
        #     model=model,
        #     optimizer=optimizer,
        #     ss_scheduler=ss_scheduler,
        #     train_dataloader=train_dataloader,
        #     server_replay_config =client_config,
        #     max_grad_norm=client_config['data_config']['train'].get('max_grad_norm', None),
        #     anneal_config=client_config['annealing_config'] if 'annealing_config' in client_config else None,
        #     num_skips_threshold=client_config['num_skips_threshold'] if 'num_skips_threshold' in client_config else -1,
        #     ignore_subtask=client_config['ignore_subtask']
        # )

        # if trainer.optimizer is not None:
        #     initial_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())

        # annealing_config = client_config['annealing_config'] if 'annealing_config' in client_config else None

        # assert 'desired_max_samples' in client_config['data_config']['train'], 'Missing \'desired_max_samples\' entry in data config parameter'
        # desired_max_samples = client_config['data_config']['train']['desired_max_samples']

        # if trainer.optimizer is not None:  # reset the optimizer state
        #     if initial_lr > 0:
        #         trainer.optimizer.param_groups[0].update({'lr': initial_lr})
        #     initial_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
        #     trainer.reset_optimizer(initial_optimizer_state, annealing_config)

        # Mark the end of setup
        end = time.time()
        client_stats['setup'] = end - begin
        print_rank(f'Client setup cost {client_stats["setup"]}s', loglevel=logging.DEBUG)               
        begin_training = end
        
        # Training begins here
        # trainer.model.train()
        # trainer.model.zero_grad()

        # Save the client batches if we want to evaluate the privacy metrics
        # apply_privacy_metrics = (False if privacy_metrics_config is None else privacy_metrics_config['apply_metrics'])

        # This is where training actually happens
        algo_payload = None

        # if semisupervision_config != None:
        #     datasets =[get_dataset(data_path, config, task, mode="train", test_only=False, data_strct=data_strcts[i], user_idx=0) for i in range(3)]
        #     algo_payload = {'algo':'FedLabels', 'data': datasets, 'iter': iteration, 'config': semisupervision_config}

        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
            fairseq_cfg.checkpoint,
            fs_trainer,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )

        max_epoch = config["fl_config"]["epoch_per_update"]
        lr = fs_trainer.get_lr()

        # a dry run on validation set to pin the memory
        valid_subsets = fairseq_cfg.dataset.valid_subset.split(",")
        if not fairseq_cfg.dataset.disable_validation:
            for subset in valid_subsets:
                print_rank('begin dry-run validation on "{}" subset'.format(subset))
                itr = fs_trainer.get_valid_iterator(subset).next_epoch_itr(
                    shuffle=False, set_dataset_epoch=False  # use a fixed valid set
                )
                if fairseq_cfg.common.tpu:
                    itr = utils.tpu_data_loader(itr)
                for _ in itr:
                    pass
        # end of dry run section

        train_meter = meters.StopwatchMeter()
        train_meter.start()

        # clear the stats for this epoch 
        fs_trainer.reset_gradient_power() 
        num_samples = 0
        training_loss = 0
        print(f"epoch_itr.next_epoch_idx: {epoch_itr.next_epoch_idx}")
        print(f"max_epoch: {max_epoch}")
        while epoch_itr.next_epoch_idx <= max_epoch:
            # break
            if lr <= fairseq_cfg.optimization.stop_min_lr:
                # logger.info(
                #     f"stopping training because current learning rate ({lr}) is smaller "
                #     "than or equal to minimum learning rate "
                #     f"(--stop-min-lr={fairseq_cfg.optimization.stop_min_lr})"
                # )
                break

            # train for one epoch
            should_stop, num_sample, loss = fairseq_train(fairseq_cfg, fs_trainer, task, epoch_itr)
            print_rank(f"Total loss for epoch {epoch_itr.next_epoch_idx - 1}: {loss}")
            num_samples += num_sample
            training_loss = loss
            if should_stop:
                break

            # only use first validation loss to update the learning rate
            # NOTE: Only one epoch in client training, remove lr step temporally
            # lr = fs_trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            epoch_itr = fs_trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=task.has_sharded_data("train"),
                # don't cache epoch iterators for sharded datasets
                disable_iterator_cache=task.has_sharded_data("train"),
            )

            '''
            with torch.no_grad():
                # experiment for payload transmitting
                for p, data in zip(fs_trainer.model.parameters(), model_parameters):
                    data = to_device(data)
                    p.grad = p.data


                for p in fs_trainer.model.parameters():
                    p.grad = num_samples * p.grad
                
                payload = {}
                payload['weight'] = num_samples
                payload['gradients'] = [p.grad.to(torch.device('cpu')) for p in fs_trainer.model.parameters()]

                for p, client_grad in zip(update_model.parameters(), payload['gradients']):
                    if p.grad is None:
                        p.grad = to_device(client_grad)
                    else:
                        p.grad += to_device(client_grad)

                torch.cuda.empty_cache()

                # Normalize with weight_sum
                for p in update_model.parameters():
                    p.grad /= payload['weight']

                for p in update_model.parameters():
                    updated_weight = p.grad
                    p.data.copy_(updated_weight)

                # update_optimizer.step()
                update_optimizer.zero_grad()

            
                cp_path = checkpoint_utils.save_checkpoint(
                    fairseq_cfg.checkpoint, fs_trainer, epoch_itr, 0
                )


                fs_trainer = fairseq_trainer(fairseq_cfg, task, model, criterion, quantizer)

                extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
                    fairseq_cfg.checkpoint,
                    fs_trainer,
                    # don't cache epoch iterators for sharded datasets
                    disable_iterator_cache=task.has_sharded_data("train"),
                )

                model_parameters = [p.data.to(torch.device('cpu')) for p in update_model.parameters()]

                for p, data in zip(fs_trainer.model.parameters(), model_parameters):
                    p.data = data.detach().clone().cuda() if torch.cuda.is_available() else data.detach().clone()
            '''

        train_meter.stop()
        fs_trainer.estimate_sufficient_stats()

        
        # logger.info("done training in {:.1f} seconds".format(train_meter.sum))

        # ioPath implementation to wait for all asynchronous file writes to complete.
        if fairseq_cfg.checkpoint.write_checkpoints_asynchronously:
            # logger.info(
            #     "ioPath PathManager waiting for all asynchronous checkpoint "
            #     "writes to finish."
            # )
            PathManager.async_close()
            # logger.info("ioPath PathManager finished waiting.")

        
        # train_loss, num_samples, algo_computation = trainer.train_desired_samples(desired_max_samples=desired_max_samples, apply_privacy_metrics=apply_privacy_metrics, algo_payload = algo_payload)
        # print_rank('client={}: training loss={}'.format(client_id[0], train_loss), loglevel=logging.DEBUG)

        # Estimate gradient magnitude mean/var
        # Now computed when the sufficient stats are updated.
        # assert 'sum' in trainer.sufficient_stats
        # assert 'mean' in trainer.sufficient_stats
        
        # trainer.train_loss = train_loss
        # trainer.num_samples = num_samples
        # trainer.algo_computation = algo_computation

        

        # Compute pseudo-gradient
        # print_rank(f"Fairseq trainer parameters: {fs_trainer.model.parameters()}")
        # print_rank(f"Model parameters: {model_parameters}")

        if not send_dicts:
            with torch.no_grad():
                for p, data in zip(fs_trainer.model.parameters(), model_parameters):
                    data = to_device(data)
                    # print_rank(f"Data: {data}")
                    # print_rank(f"P data: {p.data}")
                    # print_rank(f"data type: {type(data[0])}")
                    # print_rank(f"p data type: {type(p.data[0])}")
                    # print_rank(data.dtype)
                    # print_rank(p.data.dtype)
                    p.grad = p.data
                    # print(f"p.grad: {p.grad}")

        print_rank(f"Success!. Send Graidents: {send_gradients}")

        payload = strategy.generate_client_payload(fs_trainer, num_samples) if send_gradients else None



        # TODO: Personalization?
        # if config['server_config']['type'] == 'personalization':
        #     # Initialize convex weight alpha
        #     alpha= config['client_config'].get('convex_model_interp', 0.75)
        #     local_model = make_model(config['model_config'])
        #     train_dataloader = make_train_dataloader(data_config, data_path, task=task, clientx=0, data_strct=data_strct)
        #     local_optimizer = make_optimizer(client_config['optimizer_config'], local_model)

        #     # Make the trainer
        #     local_trainer = Trainer(
        #         model=local_model,
        #         optimizer=local_optimizer,
        #         ss_scheduler=ss_scheduler,
        #         train_dataloader=train_dataloader,
        #         server_replay_config=client_config,
        #         max_grad_norm=client_config['data_config']['train'].get('max_grad_norm', None),
        #         anneal_config=client_config['annealing_config'] if 'annealing_config' in client_config else None,
        #         num_skips_threshold=client_config[
        #             'num_skips_threshold'] if 'num_skips_threshold' in client_config else -1,
        #         ignore_subtask=client_config['ignore_subtask']
        #     )

        #     local_model_name = os.path.join(model_path, user + '_model.tar')
        #     local_alpha_name = os.path.join(model_path, user + '_alpha')

        #     if os.path.exists(local_model_name):
        #         print_rank('Loading Local Model .. {}'.format(local_model_name))
        #         local_trainer.load(local_model_name, update_lr_scheduler=False, update_ss_scheduler=False)

        #     if os.path.exists(local_alpha_name):
        #         print_rank('Loading Alpha Weight .. {}'.format(local_model_name), loglevel=logging.INFO)
        #         alpha = torch.load(local_alpha_name)

        #     # Copy original model
        #     original_local_model = local_trainer.get_model()

        #     # Training begins here
        #     local_trainer.model.train()
        #     local_trainer.model.zero_grad()

        #     # Run Local Processing
        #     train_loss, num_samples = local_trainer.train_desired_samples(desired_max_samples=desired_max_samples,
        #                                                                   apply_privacy_metrics=False)
        #     print_rank('client={}, user:{}: LOCAL training loss={}'.format(client_id[0], user, train_loss), loglevel=logging.INFO)

        #     local_trainer.save(
        #         model_path=model_path,
        #         config=config,
        #         token=user)

        #     # Estimate the pseudo-gradient for local model
        #     for p, orig_param in zip(local_trainer.model.parameters(), original_local_model.parameters()):
        #         orig_param = orig_param.cuda() if torch.cuda.is_available() else orig_param
        #         p.grad = orig_param.data - p.data

        #     alpha= alpha_update(local_trainer.model, trainer.model, alpha, initial_lr)
        #     torch.save(alpha, local_alpha_name)
        #     local_trainer.model.zero_grad()


        # Mark that training (including post-processing) is finished
        end = time.time()
        client_stats['training'] = end - begin_training
        client_stats['full cost'] = end - begin
        print_rank(f'Client training cost {end - begin_training}s', loglevel=logging.DEBUG)      
        print_rank(f'Client full cost {end - begin}s', loglevel=logging.DEBUG)

        # Create dictionary that is sent back to server
        client_output = {
            'cs': client_stats, 
            'tl': training_loss, 
            'mg': fs_trainer.sufficient_stats['mag'],
            'vg': fs_trainer.sufficient_stats['var'],
            'ng': fs_trainer.sufficient_stats['mean'],
            'rg': fs_trainer.sufficient_stats['norm'],
            'ns': num_samples,
            'pl': payload,
        }
       
        # Apply privacy metrics
        # if privacy_metrics_config and privacy_metrics_config['apply_metrics']:
        #     print_rank('Applying privacy metrics', loglevel=logging.DEBUG)

        #     privacy_stats = {'Dropped clients': 0}
        #     batches = trainer.cached_batches
        #     trainer.cached_batches = []
        #     gradients = extensions.privacy.unroll_network(model.named_parameters(), select_grad=True)[0]

        #     if privacy_metrics_config['apply_indices_extraction']:
        #         allowed_word_rank = privacy_metrics_config.get('allowed_word_rank', 9000)
        #         embed_dim, vocab_size = model_config['embed_dim'], model_config['vocab_size']
        #         overlap, indices = privacy_metrics.extract_indices_from_embeddings(gradients, batches, embed_dim, vocab_size)

        #         max_overlap =  privacy_metrics_config.get('max_allowed_overlap', None)
        #         if max_overlap is not None and overlap > max_overlap:
        #             print_rank('Removing this client because we extracted {}% words and the maximum allowed is {}%'.format(overlap * 100, max_overlap * 100))
        #             client_output['wt'] = 0.0
        #             privacy_stats['Dropped clients'] = 1

        #         privacy_stats['Extracted indices percentage'] = overlap
        #         privacy_stats['Words percentage above ' + str(allowed_word_rank) + ' word rank'] = (indices > allowed_word_rank).mean() if len(indices)>0 else 0
          
        #     if privacy_metrics_config['apply_leakage_metric']:
        #         print_rank('Applying leakage metric', loglevel=logging.DEBUG)

        #         orig_params = {n: p for (n, _), p in zip(trainer.model.named_parameters(), model_parameters)}
        #         max_ratio = np.exp(privacy_metrics_config['max_leakage'])
        #         optim_config = privacy_metrics_config['attacker_optimizer_config']
        #         is_leakage_weighted = privacy_metrics_config['is_leakage_weighted']

        #         leakage = privacy_metrics.practical_epsilon_leakage(orig_params,
        #             trainer.model, batches, is_leakage_weighted, max_ratio, optim_config)                
        #         print_rank('privacy leakage: {}'.format(leakage), loglevel=logging.DEBUG)

        #         max_leakage =  privacy_metrics_config.get('max_allowed_leakage', None)
        #         if max_leakage is not None and leakage > max_leakage:
        #             print_rank('Removing this client because the information leakage/practical epsilon is {} and the maximum allowed is {}'.format(leakage, max_leakage))
        #             client_output['wt'] = 0.0
        #             privacy_stats['Dropped clients'] = 1

        #         privacy_stats['Practical epsilon (Max leakage)'] = leakage
            
        #     client_output['ps'] = privacy_stats

        client_output['ts'] = time.time()

        # print_rank(f"client_output: {client_output}")
        return client_output
