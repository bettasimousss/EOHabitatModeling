import argparse

import os
import rasterio as rio
import tifffile as tiff

import numpy as np
import pandas as pd

PROJ_WORK_DIR = '.'
PROJ_SCRATCH_DIR = '.'

s2_rgbnir_mosaic = '%s/eo_mosaics/S2_RGBNIR.vrt'%PROJ_SCRATCH_DIR
s2_swir_mosaic = '%s/eo_mosaics/S2_SWIR.vrt'%PROJ_SCRATCH_DIR
s1_mosaic = '%s/eo_mosaics/S1.vrt'%PROJ_SCRATCH_DIR
dem_mosaic = '%s/eo_mosaics/DEM.vrt'%PROJ_SCRATCH_DIR

def extract_patch_values(src_file, lon, lat, N, do_translate=False):
    R = N//2
    mosaic_src = rio.open(src_file)
    py,px = mosaic_src.index(x=lon,y=lat)

    ## get window bounds
    wind = rio.windows.Window(max(0,px - R), max(0,py - R), N, N)
    ## extract window values
    try:
        patch = mosaic_src.read(window=wind)
        if do_translate:
            patch = patch * np.expand_dims(mosaic_src.scales,[1,2]) + np.expand_dims(mosaic_src.offsets,[1,2])
            
    except ValueError:
        print('Error extracting patch for coordinates (%.2f, %.2f)'%(lon,lat))

    return patch

batch_size = 10000

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Extract satellite Patches", add_help=True)

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
    os.makedirs('%s/dem'%img_folder,exist_ok=True)
    os.makedirs('%s/s1'%img_folder,exist_ok=True)
    os.makedirs('%s/s2'%img_folder,exist_ok=True)

    print('Loading data file: %s'%file_path)
    obs_data = pd.read_parquet(file_path).reset_index(drop=True)

    print('Starting extraction - size: ',obs_data.shape,', batch: ',batch,', start: ',start,', end: ',end)

    print(obs_data.loc[start:end,[idxvar,xvar,yvar]].head())
    
    cpt = 0
    for index, lon, lat in obs_data.loc[start:end,[idxvar,xvar,yvar]].values:
        if cpt%500==0:
            print('Processed %d samples'%cpt)

        if os.path.exists('%s/dem/%s_alti.tif'%(img_folder,index))==False:
            try:
                dem_patch = extract_patch_values(dem_mosaic, lon, lat, img_size, do_translate=True)    
                tiff.imwrite('%s/dem/%s_alti.tif'%(img_folder,index), dem_patch)
            except:
                print('Error - DEM - plot %d'%cpt)
                pass
        else:
            print('%s already done !'%index)
            
        if os.path.exists('%s/s1/%s_s1.tif'%(img_folder,index))==False:
            try:
                s1_patch = extract_patch_values(s1_mosaic, lon, lat, img_size, do_translate=True)    
                tiff.imwrite('%s/s1/%s_s1.tif'%(img_folder,index), s1_patch)
            except:
                print('Error - sentinel 1 - plot %d'%cpt)
                pass
        else:
            print('%s already done !'%index)
            
        if os.path.exists('%s/s2/%s_s2.tif'%(img_folder,index))==False:
            try:
                rgbnir_patch = extract_patch_values(s2_rgbnir_mosaic, lon, lat, img_size, do_translate=True)
                swir_patch = extract_patch_values(s2_swir_mosaic, lon, lat, img_size//2, do_translate=True)
                swir_patch2 = np.repeat(np.repeat(swir_patch, 2, axis=1), 2, axis=2)
                del swir_patch
                s2_patch = np.concatenate([rgbnir_patch[[2,1,0,3],:,:],swir_patch2],axis=0)
                del rgbnir_patch, swir_patch2
                tiff.imwrite('%s/s2/%s_s2.tif'%(img_folder,index), s2_patch)
            except:
                print('Error - sentinel 2 - plot %d'%cpt)
                pass
                
        else:
            print('%s already done !'%index)

        cpt+=1
