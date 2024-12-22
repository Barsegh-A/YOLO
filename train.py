import json
import argparse

import torch
import torchvision
import albumentations

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from src import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    
    parser.add_argument('--config_file_path', type=str, default='./configs/config_resnet18.json',
                        help='Path to the experiment config')
    

    args = parser.parse_args()

    return args

def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        config = json.load(f)

    return config

def main(args):

    config = load_config(args.config_file_path)
    model_config = config['model_config']

    
    yolo_model = YOLOV1(model_config)
    criterion = YOLOLoss(S=model_config['S'], B=model_config['B'], C=model_config['C'])


    train_dataset = VOCDataset(image_set='train', transform=transforms_alb['train'])
    val_dataset = VOCDataset(image_set='val', transform=transforms_alb['val'])


    lightning_model = YOLOV1LightningModule(model=yolo_model, criterion=criterion, learning_rate=config['learning_rate'])
    data_module = VOCDataModule(train_dataset, val_dataset, batch_size=config['batch_size'])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="yolo-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    mlf_logger = MLFlowLogger(experiment_name=config['experiment_name'], log_model=True, run_name=config['run_name'])

    mlf_logger.log_hyperparams(config)

    trainer = Trainer(
        max_epochs=config['n_epochs'],
        accelerator='gpu',
        devices=1,
        # precision='bf16-mixed',
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        logger=mlf_logger
    )

    trainer.fit(lightning_model, data_module)

if __name__ == "__main__":
    args = parse_args()
    main(args)