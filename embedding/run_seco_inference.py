import argparse
import sys
import os

import glob
import yaml

import pandas as pd
import numpy as np

import tifffile
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

sys.path.append('../foundation_models')
from SecoEco.downstream_tasks.models.moco2_module_multiband import MocoV2

SCRATCH_DIR = '.'
img_dir = '%s/habitat_images/'%SCRATCH_DIR

global_database = pd.read_parquet('dataset/habitat_database_v3.parquet',columns=['PlotID'])

####################################################################################################################################################
# ## DATASETS
# ###################################################################################################################################################
sentinel2_bands = {
    "B01": {
        "band_name": "Coastal Aerosol (Band 1)",
        "wavelength": "443 nm",
        "resolution": "60 meters",
        "application": "Aerosol detection, especially in coastal and inland waters.",
        "description": "Useful for detecting aerosols and observing water bodies."
    },
    "B02": {
        "band_name": "Blue (Band 2)",
        "wavelength": "490 nm",
        "resolution": "10 meters",
        "application": "True-color imagery and aquatic vegetation monitoring.",
        "description": "Captures visible blue light, essential for generating true-color images."
    },
    "B03": {
        "band_name": "Green (Band 3)",
        "wavelength": "560 nm",
        "resolution": "10 meters",
        "application": "True-color imagery and vegetation vigor measurement.",
        "description": "Captures visible green light, crucial for monitoring vegetation health."
    },
    "B04": {
        "band_name": "Red (Band 4)",
        "wavelength": "665 nm",
        "resolution": "10 meters",
        "application": "Vegetation monitoring and NDVI calculation.",
        "description": "Captures visible red light, important for assessing plant health."
    },
    "B05": {
        "band_name": "Vegetation Red Edge (Band 5)",
        "wavelength": "705 nm",
        "resolution": "20 meters",
        "application": "Vegetation analysis, particularly chlorophyll content.",
        "description": "Sensitive to changes in vegetation, useful for detecting plant stress."
    },
    "B06": {
        "band_name": "Vegetation Red Edge (Band 6)",
        "wavelength": "740 nm",
        "resolution": "20 meters",
        "application": "Vegetation analysis and stress detection.",
        "description": "Helps in detailed vegetation monitoring, especially under stress conditions."
    },
    "B07": {
        "band_name": "Vegetation Red Edge (Band 7)",
        "wavelength": "783 nm",
        "resolution": "20 meters",
        "application": "Plant health monitoring and water content assessment.",
        "description": "Useful for observing the health and water content of vegetation."
    },
    "B08": {
        "band_name": "Near Infrared (NIR) (Band 8)",
        "wavelength": "842 nm",
        "resolution": "10 meters",
        "application": "NDVI and other vegetation indices.",
        "description": "Highly reflective in healthy vegetation, critical for vegetation indices."
    },
    "B08A": {
        "band_name": "Narrow NIR (Band 8A)",
        "wavelength": "865 nm",
        "resolution": "20 meters",
        "application": "Vegetation health, particularly in high chlorophyll areas.",
        "description": "Provides additional details on vegetation health."
    },
    "B09": {
        "band_name": "Water Vapour (Band 9)",
        "wavelength": "945 nm",
        "resolution": "60 meters",
        "application": "Water vapor content analysis in the atmosphere.",
        "description": "Measures water vapor, useful for atmospheric studies."
    },
    "B10": {
        "band_name": "SWIR Cirrus (Band 10)",
        "wavelength": "1375 nm",
        "resolution": "60 meters",
        "application": "Cirrus cloud detection.",
        "description": "Designed specifically for detecting cirrus clouds in the atmosphere."
    },
    "B11": {
        "band_name": "SWIR (Band 11)",
        "wavelength": "1610 nm",
        "resolution": "20 meters",
        "application": "Water content in soil and vegetation, snow/ice detection.",
        "description": "Sensitive to water content, useful for monitoring snow and ice."
    },
    "B12": {
        "band_name": "SWIR (Band 12)",
        "wavelength": "2190 nm",
        "resolution": "20 meters",
        "application": "Surface material differentiation (e.g., snow, ice, clouds).",
        "description": "Useful for distinguishing between different surface types and monitoring clouds."
    }
}

def interpolate_band(band_low, band_high, lambda_low, lambda_high, lambda_missing):
    """
    Interpolates between two bands to estimate a missing band.

    Parameters:
        band_low (numpy.ndarray): The lower wavelength band.
        band_high (numpy.ndarray): The higher wavelength band.
        lambda_low (float): The wavelength of the lower band.
        lambda_high (float): The wavelength of the higher band.
        lambda_missing (float): The wavelength of the missing band.

    Returns:
        numpy.ndarray: The interpolated band.
    """
    return band_low + ((lambda_missing - lambda_low) / (lambda_high - lambda_low)) * (band_high - band_low)

def complete_sentinel_image_channel_first(input_image,k_band1 = 0.9,k_band8a = 1.05):
    """
    Completes a Sentinel-2 image with missing bands for channel-first format.

    Parameters:
        input_image (numpy.ndarray): Input image with shape (num_bands, height, width).
                                     Assumed bands are in the order ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'].

    Returns:
        numpy.ndarray: Complete image with shape (13, height, width) including interpolated and approximated bands.
    """
    # Ensure the input image has 6 bands
    if input_image.shape[0] != 6:
        raise ValueError("Input image must have exactly 6 bands.")

    num_bands, height, width = input_image.shape
    complete_image = np.zeros((13, height, width))

    # Assign available bands to their positions in the complete image
    complete_image[1] = input_image[0]  # Blue (B02)
    complete_image[2] = input_image[1]  # Green (B03)
    complete_image[3] = input_image[2]  # Red (B04)
    complete_image[7] = input_image[3]  # NIR (B08)
    complete_image[11] = input_image[4]  # SWIR1 (B11)
    complete_image[12] = input_image[5]  # SWIR2 (B12)

    # Approximate missing bands
    # Band 1 (Coastal/Aerosol, 443 nm) from Blue (490 nm) using a simple scaling factor
    complete_image[0] = k_band1 * input_image[0]  # Approximating Band 1

    # Band 5 (Red Edge 1, 705 nm) between Red (665 nm) and NIR (842 nm)
    complete_image[4] = interpolate_band(input_image[2], input_image[3], 665, 842, 705)

    # Band 6 (Red Edge 2, 740 nm) between Red (665 nm) and NIR (842 nm)
    complete_image[5] = interpolate_band(input_image[2], input_image[3], 665, 842, 740)

    # Band 7 (Red Edge 3, 783 nm) between Red (665 nm) and NIR (842 nm)
    complete_image[6] = interpolate_band(input_image[2], input_image[3], 665, 842, 783)

    # Band 8A (Narrow NIR, 865 nm) approximated from NIR (842 nm)
    complete_image[8] = k_band8a * input_image[3]

    # Bands 9 (Water Vapor, 945 nm) and 10 (Cirrus, 1375 nm) are missing and set to zero
    complete_image[9] = np.zeros((height, width))  # Water Vapor
    complete_image[10] = np.zeros((height, width))  # Cirrus

    return complete_image

def calculate_ndvi(nir_band, red_band):
    nir_band = nir_band
    red_band = red_band
    ndvi = (nir_band - red_band) / (nir_band + red_band + 0.00001)
    ndvi[ndvi < 0] = 0
    return ndvi

class SecoEcoDataset(Dataset):
    def __init__(self, img_dir, imsize):
        self.img_dir = img_dir
        self.img_list = glob.glob('%s/*.tif'%img_dir)
        self.imsize = imsize
        self.transform = transforms.Compose([
            transforms.CenterCrop(imsize),
            transforms.Resize((224, 224))
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        filename = self.img_list[idx]
        image = tifffile.imread(filename)
        ## Complete missing bands
        image = complete_sentinel_image_channel_first(image)
        ## Get Seco-Eco bands only
        image = image[SE_idx_bands,:,:]
        ## Compute NDVI
        ndvi_band = np.expand_dims(calculate_ndvi(nir_band=image[nir_idx,:,:], red_band=image[red_idx,:,:]),axis=0)
        ## Add to image
        image = np.concatenate([image,ndvi_band],axis=0)
        
        ## Multiply by 10000
        image = torch.FloatTensor(image) #(image*10000).round(decimals=0))
        image = self.transform(image)
        
        return filename, image

####################################################################################################################################################
# ## CALLBACKS
# ###################################################################################################################################################
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
class SecoEcoInferenceModule(pl.LightningModule):
    def __init__(self, checkpoint):
        super().__init__()

        ### Build architecture
        checkpoint = MocoV2.load_from_checkpoint(seco_checkpoint,arch="resnet50", emb_dim=128, moco_k=65536, bands = 'B9')
        self.model = copy.deepcopy(checkpoint.encoder_q)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    def forward(self, x):
        file, image = x
        return self.model(image)

####################################################################################################################################################
# ## CONSTANTS
# ################################################################################################################################################## 
seed = 1234

all_bands = list(sentinel2_bands.keys())
available_bands = ['B02','B03','B04','B08','B11','B12']

SE_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'NDVI']
SE_idx_bands = [i for i, x in enumerate(all_bands) if x in SE_BANDS]
nir_idx = 6 #B08
red_idx = 2 #B04

seco_checkpoint = '../foundation_models/SecoEco/checkpoint/seco-eco_e100.ckpt'
output_dir = '%s/habitat_embeddings'%SCRATCH_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser("SSL4ECO run inference", add_help=True)
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
        help="Image size",
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

    ### Dataset, DataLoader
    seco_sentinel_dataset = seco_dataset = SecoEcoDataset(img_dir=data_dir,imsize=imsize)
    seco_sentinel_dl = DataLoader(seco_dataset,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers = num_workers)
    
    # Use pre-trained model weights
    seco_model = SecoEcoInferenceModule(checkpoint=seco_checkpoint)
    
    # Callback
    out_dir = '%s/seco_s2'%output_dir
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
    trainer.predict(seco_model, seco_sentinel_dl,return_predictions=False)
    
    # Confirm end
    print('END OF INFERENCE')
    
