import argparse
import sys
import os

import glob
import yaml

import pandas as pd
import numpy as np

import tifffile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

sys.path.append('../foundation_models')

SCRATCH_DIR = '.'
img_dir = '%s/habitat_images/'%SCRATCH_DIR

global_database = pd.read_parquet('dataset/habitat_database_v3.parquet',columns=['PlotID'])

####################################################################################################################################################
# ## DATASETS
# ###################################################################################################################################################
class PrithviDataset(Dataset):
    def __init__(self, img_dir, imsize, mean, std):
        self.img_dir = img_dir
        self.img_list = glob.glob('%s/*.tif'%img_dir)

        self.transform = transforms.Compose([
            transforms.CenterCrop(128),  
            transforms.Resize((224, 224)),  
            transforms.Normalize(mean=mean,std=std)
        ])
        
        self.imsize = imsize

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        filename = self.img_list[idx]
        image = tifffile.imread(filename)
        image = torch.FloatTensor(image)
        
        if self.transform:
            image = self.transform(image*10000)
        
        return filename, image

class CustomWriter(pl.callbacks.BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
    
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        filenames = batch[0]
        unique_batch_id = f"rank_{trainer.global_rank}_dataloader_{dataloader_idx}_batch_{batch_idx}"
        torch.save([filenames,prediction], os.path.join(self.output_dir,f"{unique_batch_id}_embeddings.pt"))
        
####################################################################################################################################################
# ## MODEL, TRAINING, CALLBACKS
# ###################################################################################################################################################  

sys.path.append('../foundation_models/Prithvi_100M')

from prithvi_mae import PrithviMAE

class PrithviInferenceModule(pl.LightningModule):
    def __init__(self, checkpoint, yaml_file_path, img_size=224, return_patch_embed=False):
        super().__init__()

        self.return_patch_embed = return_patch_embed
        self.img_size = img_size
        
        ### Load weights
        prithvi_checkpoint = torch.load(checkpoint, map_location="cpu")
        del prithvi_checkpoint['encoder.pos_embed']

        ### Load config file
        with open(yaml_file_path) as f:
            prithvi_config = yaml.safe_load(f)

        model_args, train_args = prithvi_config["model_args"], prithvi_config["train_params"]
        model_args["num_frames"] = 1
        model_args["img_size"] = img_size

        self.model = PrithviMAE(encoder_only=True,**model_args)
        self.model.load_state_dict(prithvi_checkpoint, strict=False)
        self.model.eval()

    def forward(self, x):
        file, image = x
        ### Get last transformer block output
        features = self.model.encoder.forward_features(image)[-1]

        if self.return_patch_embed:
            features = features[:,1:,:]
        else:
            ### Get class token (image-level embedding)
            features = features[:,0,:]
        return features

####################################################################################################################################################
# ## CONSTANTS
# ################################################################################################################################################### 

seed = 1234
prithvi_yaml_file_path = '../foundation_models/Prithvi_100M/config.yaml'
prithvi_weights_path = '../foundation_models/Prithvi_100M/Prithvi_EO_V1_100M.pt'

with open(prithvi_yaml_file_path) as f:
    prithvi_config = yaml.safe_load(f)

_, train_args = prithvi_config["model_args"], prithvi_config["train_params"]

output_dir = '%s/habitat_embeddings'%SCRATCH_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prithvi run inference", add_help=True)
    parser.add_argument(
        "--batch",
        default=128,
        type=int,
        help="Batch number",
    )      

    parser.add_argument(
        "--workers",
        default=10,
        type=int,
        help="Number of workers",
    )  

    parser.add_argument(
        "--imsize",
        type=int,
        help="Image crop size",
    )     

    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.device_count())  # Should match your total GPUs
    
    args = parser.parse_args()
    
    ### ARGUMENTS
    batch_size = args.batch
    num_workers = args.workers
    imsize = args.imsize
    data_dir = '%s/s2'%img_dir
    
    ### START INFERENCE
    pl.seed_everything(seed, workers=True) 
    pid_list = global_database['PlotID'].tolist()

    ### Dataset, DataLoader
    prithvi_sentinel_dataset = PrithviDataset(img_dir=data_dir,imsize=imsize,
                                              mean=train_args['data_mean'],std=train_args['data_std'])
    
    prithvi_sentinel_dl = DataLoader(prithvi_sentinel_dataset,batch_size=batch_size,shuffle=False,
                                     pin_memory=True,num_workers = num_workers)
    
    # Use pre-trained model weights
    prithvi_model = PrithviInferenceModule(prithvi_weights_path, prithvi_yaml_file_path,
                                           return_patch_embed=False)
    
    # Callback
    out_dir = '%s/prithvi_s2'%output_dir
    os.makedirs(out_dir,exist_ok=True)
    pred_writer = CustomWriter(output_dir=out_dir, write_interval="batch")
    
    # Start prediction
    trainer_args = {'accelerator': 'gpu',
                    'devices': int(os.environ['SLURM_GPUS_ON_NODE']),
                    'num_nodes': int(os.environ['SLURM_NNODES']),
                    'strategy': 'auto'
                   }

    print(trainer_args)
    
    trainer = pl.Trainer(**trainer_args,callbacks=[pred_writer])
    trainer.predict(prithvi_model, prithvi_sentinel_dl,return_predictions=False)
    
    # Confirm end
    print('END OF INFERENCE')
    
