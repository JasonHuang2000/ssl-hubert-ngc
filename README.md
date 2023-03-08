# Requirements and Installation

* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/willymaster7749/ssl-hubert.git
cd ssl-hubert
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **To install dependencies for FLUTE**:

``` bash
git clone https://github.com/willymaster7749/ssl-hubert.git
cd ssl-hubert/flute
pip install -r requirements.txt
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Run HuBERT pre-training with Fariseq & FLUTE

1. Add fairseq related arguments (like the example in **```ssl-hubert/hubert_pretrain.sh```**) to **```ssl-hubert/fairseq_cli/hydra_train.py```**, in **```main```**:
```python
args = Namespace(help=False, hydra_help=False, 
        overrides=['task.data=/path/librispeech/LibriSpeech/dev-clean/manifest', 
        'task.label_dir=/path/librispeech/LibriSpeech/dev-clean/labels', 
        'task.labels=["km"]', 'model.label_rate=100'], cfg=None, package=None, run=False, 
        multirun=False, shell_completion=False, config_path=None, 
        config_name='hubert_base_librispeech', 
        config_dir='/path/fairseq/examples/hubert/config/pretrain', 
        info=False)
```

2. Modify **```_run_hydra```** function in **```hydra```** python site-package, **```{python-site-packages}/hydra/_internal/utils.py```**: 

```python
def _run_hydra(
    args_parser: argparse.ArgumentParser,
    task_function: TaskFunction,
    config_path: Optional[str],
    config_name: Optional[str],
    strict: Optional[bool],
    args, # add this line
) -> None:

    from hydra.core.global_hydra import GlobalHydra

    from .hydra import Hydra
    
    # comment the following line
    # args = args_parser.parse_args()
    if args.config_name is not None:
        config_name = args.config_name
```

3. Run the experiment. 
- In **```ssl-hubert/flute/fl_hubert_pretrain.sh```**
    - **nproc_per_node**: Number of process, including one server process and several client process. (If set to 5, there will be a server process and 4 client process)
    - **config**: The config file for FL experiment

```bash
cd ssl-hubert/flute
bash fl_hubert_pretrain.sh
```

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

