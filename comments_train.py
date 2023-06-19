# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch 

from models.unet_lightning import UNet_Lightning as UNetModel
from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData

class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params     
        self.training_params = training_params
        if mode in ['train']:
            print("Loading TRAINING/VALIDATION dataset -- as test")
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
            print(f"Training dataset size: {len(self.train_ds)}")
        if mode in ['val']:
            print("Loading VALIDATION dataset -- as test")
            self.val_ds = RainData('validation', **self.params)  
        if mode in ['predict']:    
            print("Loading PREDICTION/TEST dataset -- as test")
            self.test_ds = RainData('test', **self.params)
        if mode in ['heldout']:    
            print("Loading HELD-OUT dataset -- as test")
            self.test_ds = RainData('heldout', **self.params)   

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        # (hr) 일반적으로, dataloader에서 pin_memory=True, num_workers=gpu개수*4로 설정하면 좋다고 조은빈님이 가르쳐줬었음
        dl = DataLoader(dataset, 
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, 
                        pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl
    
    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)
    
    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(Model, params, checkpoint_path=''):
    # (hr) 입력 인자 중 model은 class이기 때문에 대문자로 시작함(Model)
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']} 
        # 딕셔너리 언패킹을 통해 params['experiment'], params['dataset'], params['train']에 있는 키-값 쌍들을 하나로 합쳐서 p 딕셔너리를 생성
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p)
            # (hr) load_from_checkpoint은 pytorch_lightning 라이브러리의 LightningModule 모듈에 있는 함수
    return model
      # (hr) return 되는 것: LightningModule instance with loaded weights and hyperparameters (if available).


def get_trainer(gpus,params):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for",max_epochs,"epochs");
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=3, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')
    
    parallel_training = None
    ddpplugin = None   
    if gpus[0] == -1:
        gpus = None
    elif len(gpus) > 1:
        parallel_training = 'ddp'
            # (hr) ddp: distributed data parallel ..인 듯
##        ddpplugin = DDPPlugin(find_unused_parameters=True)
    print(f"====== process started on the following GPUs: {gpus} ======")
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params['experiment']['name']
    version = version + '_' + date_time

    #SET LOGGER 
    if params['experiment']['logging']: 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],name=params['experiment']['sub_folder'], version=version, log_graph=True)
    else: 
        tb_logger = False

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]

    trainer = pl.Trainer(devices=gpus, max_epochs=max_epochs,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         accelerator="gpu",
                         callbacks=callback_funcs,logger=tb_logger,
                         profiler='simple',
                             # (hr) profiler: To profile individual steps during training and assist in identifying bottlenecks. Default: None.
                         precision=params['experiment']['precision'],
                         strategy="ddp"
                             # (hr) strategy: Supports different training strategies with aliases as well custom strategies. Default: "auto".
                        )

    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
      # (hr) predict: pytorch_lightningd의 Trainer 클래스의 method.
      #               Run inference on your data. This will call the model forward function to compute predictions. 
      #               Useful to perform distributed and batched predictions. Logging is disabled in the predict hooks.
    scores = torch.concat(scores)   
    tensor_to_submission_file(scores,predict_params)
      # (hr) saves prediction tesnor to submission .h5 file
      #      submission_out_dir 아래에 year 아래에 region.pred.h5 이름으로 저장됨

def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)
      # (hr) test: pytorch_lightningd의 Trainer 클래스의 method.
      #            Perform one evaluation epoch over the test set. 
      #            It’s separated from fit to make sure you never run on your test set until you want to.
    
def train(params, gpus, mode, checkpoint_path, model=UNetModel): 
    """ main training/evaluation method
    """
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
        # (hr) dataset, train은 config yaml 파일에 있음. mode는 코드 수행 시 지정된 값
    model = load_model(model, params, checkpoint_path)
    # ------------
    # Add your models here
    # ------------
    
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params)
      # (hr) trainer는 pytorch_lightning의 Trainer 클래스 인스턴스
    get_cuda_memory_usage(gpus)
    # ------------
    # train & final validation
    # ------------
    if mode == 'train':
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)
    
    
    if mode == "val":
    # ------------
    # VALIDATE
    # ------------
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_test(trainer, model, data.val_dataloader()) 


    if mode == 'predict' or mode == 'heldout':
    # ------------
    # PREDICT
    # ------------
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        print("REGIONS!:: ", params["dataset"]["regions"], params["predict"]["region_to_predict"])
        if params["predict"]["region_to_predict"] not in params["dataset"]["regions"]:
            print("EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            do_predict(trainer, model, params["predict"], data.test_dataloader())
              # (hr) mode가 predict인지 heldout인지에 따라 test_dataloader()에 load 되어있는 데이터가 다름
    
    get_cuda_memory_usage(gpus)

def update_params_based_on_args(options):
    config_p = os.path.join('models/configurations',options.config_path)
    params = load_config(config_p)
    
    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configurations/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1, 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='', 
                         help="Set the name of the experiment")

    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)
    train(params, options.gpus, options.mode, options.checkpoint)

if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py --gpus 2 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    6) generate predictions for the held-out dataset (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode heldout  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"
        # (hr) held-out dataset은 test set을 뜻함
    """
