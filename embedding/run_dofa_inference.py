import argparse
import sys
import os

import glob

import pandas as pd
import numpy as np

import tifffile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

sys.path.append('../foundation_models/DOFA')

from models_dwv import vit_base_patch16

SCRATCH_DIR = '.'
img_dir = '%s/habitat_images/'%SCRATCH_DIR

global_database = pd.read_parquet('dataset/habitat_database_v3.parquet',columns=['PlotID'])

####################################################################################################################################################
### DATASETS
####################################################################################################################################################
class DofaS1Dataset(Dataset):
    def __init__(self, img_dir, imsize, mean=None, std=None, scale=0.001, offset=-45, max_range=65535):
        self.img_dir = img_dir
        self.img_list = glob.glob('%s/*.tif'%img_dir)

        self.scale = scale
        self.offset = offset
        self.max_range = max_range

        self.transform = transforms.Compose([
            transforms.CenterCrop(imsize),
            transforms.Resize((224, 224)),  
            transforms.Normalize(mean=mean,std=std)
        ])
            
        self.imsize = imsize

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        filename = self.img_list[idx]
        image = tifffile.imread(filename)
        image = torch.FloatTensor(image[0:2,:,:])

        if self.transform:
            ### reverse scaling / offset > image in [0:65535] range
            image = (image - self.offset) / self.scale
    
            ### transform to an image in [0:1] range
            image = image / self.max_range
            
            ### Multiply by 255 to align with DOFA pretraining range
            image = self.transform(image*255)
        
        return filename, image

class DofaS2Dataset(Dataset):
    def __init__(self, img_dir, imsize, mean=None, std=None):
        self.img_dir = img_dir
        self.img_list = glob.glob('%s/*.tif'%img_dir)

        self.transform = self.transform = transforms.Compose([
            transforms.CenterCrop(imsize),
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
            image = self.transform(image*255)
        
        return filename, image

class DofaSentinelDataset(Dataset):
    def __init__(self, img_dir, pid_list, imsize, s1_mean=None, s1_std=None, s2_mean=None, s2_std=None, scale=0.001, offset=-45, max_range=65535):
        self.img_dir = img_dir
        self.pid_list = pid_list
        # self.s1_img_list = ['%s/s1/%s_s1.tif'%(img_dir,pid) for pid in pid_list]
        # self.s2_img_list = ['%s/s2/%s_s2.tif'%(img_dir,pid) for pid in pid_list]

        self.scale = scale
        self.offset = offset
        self.max_range = max_range        

        self.s1_transform = transforms.Compose([
            transforms.CenterCrop(imsize),
            transforms.Resize((224, 224)),  
            transforms.Normalize(mean=s1_mean,std=s1_std)
        ])
        
        self.s2_transform = transforms.Compose([
            transforms.CenterCrop(imsize),
            transforms.Resize((224, 224)),  
            transforms.Normalize(mean=s2_mean,std=s2_std)
        ])
            
        self.imsize = imsize

    def __len__(self):
        return len(self.pid_list)

    def __getitem__(self, idx):
        pid = self.pid_list[idx]
        
        s1_filename = '%s/s1/%s_s1.tif'%(self.img_dir,pid)
        s1_image = tifffile.imread(s1_filename)
        s1_image = torch.FloatTensor(s1_image[0:2,:,:])

        s2_filename = '%s/s2/%s_s2.tif'%(self.img_dir,pid)
        s2_image = tifffile.imread(s2_filename)
        s2_image = torch.FloatTensor(s2_image) 

        if self.s1_transform:
            ### reverse scaling / offset > image in [0:65535] range
            s1_image = (s1_image - self.offset) / self.scale
            ### transform to an image in [0:1] range
            s1_image = s1_image / self.max_range        
            ### Align with DOFA training
            s1_image = self.s1_transform(s1_image*255)

        if self.s2_transform:
            s2_image = self.s2_transform(s2_image*255)

        image = torch.cat([s1_image,s2_image],0)
        
        return pid, image

####################################################################################################################################################
### MODEL, TRAINING
####################################################################################################################################################        
class DOFAInferenceModule(pl.LightningModule):
    def __init__(self, checkpoint, wavelengths):
        super().__init__()
        self.wavelengths = wavelengths
        self.model = vit_base_patch16()
        state_dict = torch.load(checkpoint,weights_only=True)
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def forward(self, x):
        file, image = x
        return self.model.forward_features(image,wave_list=self.wavelengths)
        
class CustomWriter(pl.callbacks.BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
    
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        filenames = batch[0]
        unique_batch_id = f"rank_{trainer.global_rank}_dataloader_{dataloader_idx}_batch_{batch_idx}"
        torch.save([filenames,prediction], os.path.join(self.output_dir,f"{unique_batch_id}_embeddings.pt"))

####################################################################################################################################################
### CONSTANTS
#################################################################################################################################################### 
dofa_s1_mean_list = [88.45542715, 166.36275909]
dofa_s1_std_list = [43.07350145, 64.83126309]
dofa_s1_wave_list = [3.75, 3.75]

dofa_s2_mean_list = [126.63977424, 114.81779093, 114.1099739, 101.435633, 72.32804172, 56.66528851]
dofa_s2_std_list = [126.63977424, 114.81779093, 114.1099739, 101.435633, 72.32804172, 56.66528851]
dofa_s2_wave_list = [0.49, 0.56, 0.665, 0.842, 1.61, 2.19]

dofa_mean_list = dofa_s1_mean_list + dofa_s2_mean_list
dofa_std_list = dofa_s1_std_list + dofa_s2_std_list

seed = 1234
#output_dir = 'dataset/embeddings'

output_dir = '%s/habitat_embeddings'%SCRATCH_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser("DOFA run inference", add_help=True)
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
        "--mode",
        type=str,
        help="Sensor mode",
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
    mode = args.mode
    imsize = args.imsize
    
    ### START INFERENCE
    pl.seed_everything(seed, workers=True) 
    pid_list = global_database['PlotID'].tolist()
    
    if mode=='s1':
        dofa_sentinel_dataset = DofaS1Dataset(img_dir='%s/s1'%img_dir, imsize=imsize, mean=dofa_s1_mean_list, std=dofa_s1_std_list)
        dofa_wave_list = dofa_s1_wave_list
    
    if mode=='s2':
        dofa_sentinel_dataset = DofaS2Dataset(img_dir='%s/s2'%img_dir, imsize=imsize, mean=dofa_s2_mean_list, std=dofa_s2_std_list)
        dofa_wave_list = dofa_s2_wave_list
        
    if mode=='s12':
        dofa_sentinel_dataset = DofaSentinelDataset(img_dir, pid_list, imsize, 
                                                    s1_mean=dofa_s1_mean_list, s1_std=dofa_s1_std_list, 
                                                    s2_mean=dofa_s2_mean_list, s2_std=dofa_s2_std_list, 
                                                    scale=0.001, offset=-45, max_range=65535)
    
        dofa_wave_list = dofa_s1_wave_list + dofa_s2_wave_list
    
    # Data loader
    dofa_sentinel_dl = DataLoader(dofa_sentinel_dataset,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers = num_workers)
    
    # Use pre-trained model weights
    dofa_sentinel_model = DOFAInferenceModule(checkpoint='../foundation_models/DOFA/checkpoints/DOFA_ViT_base_e100.pth',wavelengths=dofa_wave_list)
    
    # Callback
    out_dir = '%s/dofa_%s'%(output_dir,mode)
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
    trainer.predict(dofa_sentinel_model, dofa_sentinel_dl,return_predictions=False)
    
    # Confirm end
    print('END OF INFERENCE')
    
