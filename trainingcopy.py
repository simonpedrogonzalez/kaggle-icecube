from graphnet.data.constants import FEATURES, TRUTH
from graphnet.training.callbacks import ProgressBar
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler
import socket

BATCH_DIR = '../input/icecube-neutrinos-in-deep-ice/train'
META_DIR = '../input/icecube-neutrinos-in-deep-ice/train'
FILTER_BY_KAPPA_THRE = 0.5

# training setting

TRAIN_MODE = False
hostName = socket.gethostname()
DROPOUT=0.0
NB_NEAREST_NEIGHBOURS = [6]
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
ONLY_AUX_FALSE = False
SERIAL_CONNECTION = True
USE_PP = True
USE_TRANS_IN_LAST=0
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


runName = 'base1-3l250p4n-batch400-650x30-infer' #TODO

    
# Configuration
FORCE_MAX_PULSE = 6000 # 強制的にこの件数以上のパルスは返さない

project = "650x20"
n_batch = 650
n_round = 10

batch_size = 400                # multi GPU
gpus = [0]                     # multi GPU
accumulate_grad_batches = {0: 5}

if len(gpus) > 1:
    if is_env_notebook():
        distribution_strategy = 'ddp_notebook'
    else:
        distribution_strategy = 'ddp'
else:
    distribution_strategy = None



config = {
        "path": 'dummy',
        "inference_database_path": '/home/tito/kaggle/icecube-neutrinos-in-deep-ice/notebook_graphnet/batch_660.db', #dummy
        "pulsemap": 'pulse_table', #dummy
        "truth_table": 'meta_table', #dummy
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": batch_size,
        "num_workers": 2, #todo
        "target": 'direction',
        "early_stopping_patience": n_batch,
        "gpus": gpus,
        "fit": {
                "max_epochs": n_batch*n_round,
                "gpus": gpus,
                "distribution_strategy": distribution_strategy,
                "check_val_every_n_epoch":10,
                "precision": 16,
                #"gradient_clip_val": 0.9,
                "reload_dataloaders_every_n_epochs": 1,
                },

        "accumulate_grad_batches": accumulate_grad_batches,
        'runName': runName,
        'project': project,
        'scheduler_verbose': False,
        'train_batch_ids': list(range(1,n_batch+1)),
        'valid_batch_ids': [660], # only suport one batch
        'test_selection': None,
        'base_dir': 'training',
        'train_len': 0,                    #not using anymore
        'valid_len': 0,                    #not using anymore
        'train_max_pulse': 300,
        'valid_max_pulse': 200,
        'train_min_pulse': 0,
        'valid_min_pulse': 0,
}


debug = False # bbb
if debug:
    runName = runName + '_debug'
    config["project"] = 'debug'
    #config["num_workers"] = 0
    config["batch_size"] = 2
    config["train_len"] = 2
    config["valid_len"] = 2


if TRAIN_MODE:

    train_dataloader, validate_dataloader, train_dataset, validate_dataset = make_dataloaders2(config = config)
    model = build_model2(config, train_dataloader, train_dataset)


    # Training model
    callbacks = [
        ModelCheckpoint(
            dirpath='../model_checkpoint_graphnet/',
            filename=runName+'-{epoch:02d}-{val_tloss:.6f}',
            monitor= 'val_tloss',
            save_top_k = 30,
            every_n_epochs = 10,
            save_weights_only=False,
        ),
        ProgressBar(),
    ]

    if 'accumulate_grad_batches' in config and len(config['accumulate_grad_batches']) > 0:
        callbacks.append(GradientAccumulationScheduler(scheduling=config['accumulate_grad_batches']))

    if debug == False:
        config["fit"]["logger"] = pl_loggers.WandbLogger(project=config["project"], name=runName)
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    #config["fit"]["profiler"] = PyTorchProfiler( output_filename='profiler_results.txt', trace_every_n_steps=1)


    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    model.save_state_dict(f'../model_graphnet/{runName}-last.pth')