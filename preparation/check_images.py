import argparse

import os
import rasterio as rio
import tifffile as tiff

import numpy as np
import pandas as pd

from tifffile import TiffFileError

PROJ_WORK_DIR = '.'
PROJ_SCRATCH_DIR = '.'

batch_size = 50000

def check_patch(patch, shape):

    is_constant = (np.nanstd(patch)==0)
    has_nan = (np.isnan(patch).sum()>0)

    has_nodata = (patch.shape!=shape)

    return is_constant, has_nan, has_nodata

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Check patches", add_help=True)

    parser.add_argument(
        "--img_folder",
        type=str,
        help="Folder to store extracted images",
    )   

    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to dataframe with lon/lat coordinates",
    )

    parser.add_argument(
        "--batch",
        default=0,
        type=int,
        help="Batch number",
    )  
    
    parser.add_argument(
        "--xvar",
        type=str,
        default='Longitude',
        help="Name of longitude column in dataframe",
    )   

    parser.add_argument(
        "--yvar",
        type=str,
        default='Latitude',
        help="Name of latitude column in dataframe",
    )    

    parser.add_argument(
        "--gridvar",
        type=str,
        default='grid',
        help="Name of plot (used in image file name) column in dataframe",
    )        
    
    parser.add_argument(
        "--img_size",
        default=512,
        type=int,
        help="Image size to be used with model. Defaults to 512",
    )   
    
    args = parser.parse_args()
    file_path = args.file_path
    img_size = args.img_size
    img_folder = args.img_folder
    
    xvar = args.xvar
    yvar = args.yvar
    idxvar = args.gridvar
    
    batch = args.batch
    
    start = batch*batch_size
    end = start+batch_size

    print('Extracting patches for plots %d to %d'%(start,end))

    os.makedirs(img_folder,exist_ok=True)
    os.makedirs('%s/s1'%img_folder,exist_ok=True)
    os.makedirs('%s/s2'%img_folder,exist_ok=True)

    print('Loading data file: %s'%file_path)
    obs_data = pd.read_parquet(file_path).reset_index(drop=True)

    print('Starting extraction - size: ',obs_data.shape,', batch: ',batch,', start: ',start,', end: ',end)
    
    cpt = 0
    diagnosis = []
    for index, lon, lat in obs_data.loc[start:end,[idxvar,xvar,yvar]].values:
        if cpt%1000==0:
            print('Processed %d samples'%cpt)
    
        empty_s1 = False
        empty_s2 = False
        
        constant_s1 = False
        constant_s2 = False
        
        hasnan_s1 = False     
        hasnan_s2 = False

        hasnodata_s1 = False
        hasnodata_s2 = False

        s1_file = '%s/s1/%s_s1.tif'%(img_folder,index)
        s2_file = '%s/s2/%s_s2.tif'%(img_folder,index)  
        
        try:
            s1_img = tiff.imread(s1_file)
        except TiffFileError:
            empty_s1 = True
            print(s1_file)
    
        try:
            s2_img = tiff.imread(s2_file)
        except TiffFileError:
            empty_s2 = True
            print(s2_file)
    
        if empty_s1==False:
            constant_s1, hasnan_s1, hasnodata_s1 = check_patch(s1_img,(3,img_size,img_size))
            del s1_img
    
        if empty_s2==False:
            constant_s2, hasnan_s2, hasnodata_s2 = check_patch(s2_img,(6,img_size,img_size))
            del s2_img
    
        out = {'pid':index,
               'empty_s1':empty_s1,'constant_s1':constant_s1,'hasnan_s1':hasnan_s1,'hasnodata_s1':hasnodata_s1,
               'empty_s2':empty_s2,'constant_s2':constant_s2,'hasnan_s2':hasnan_s2, 'hasnodata_s2':hasnodata_s2}

        diagnosis.append(out)
        cpt+=1

    full_diagnosis = pd.DataFrame.from_dict(diagnosis)
    full_diagnosis.to_csv('%s/diagnosis_%d_%d.csv'%(img_folder,start,end))    
