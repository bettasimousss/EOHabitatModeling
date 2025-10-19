import argparse
import os
import json

import numpy as np
import pandas as pd

from source.habitat_dataset import *
from source.habitat_model import *
from source.decision_forest_habitat_models import *

work_dir = 'habitat_models'
path_hab_data = 'dataset/habitat_database_v3.parquet'
path_pred_data = 'dataset/predictor_database.parquet'
path_metadata = 'dataset/feature_metadata.csv'
path_embeddings = 'viz/'
config_path = 'configs/best_xgboost.json'

plot_id = 'PlotID'
group_id = 'EUNIS1'
class_id = 'EUNIS3'

seed = 1234 
n_splits = 5

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train global habitat model", add_help=True)
    parser.add_argument(
        "--featset",
        type=str,
        help="Environmental feature sets",
    )  

    parser.add_argument(
        "--target",
        default='all',
        type=str,
        help="EUNIS formation targeted",
    )   

    parser.add_argument(
        "--imsize",
        type=int,
        help="Landscape image size",
    )       

    parser.add_argument(
        "--msi",
        default='none',
        type=str,
        help="Foundation model for MSI data",
    ) 

    parser.add_argument(
        "--sar",
        default='none',
        type=str,
        help="Foundation model for SAR data",
    ) 

    parser.add_argument(
        "--ncomp",
        default=100,
        type=int,
        help="Embedding crop size",
    )     

    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="Device",
    )    

    parser.add_argument(
        "--dirpath",
        default="none",
        type=str,
        help="Device",
    )        

    args = parser.parse_args()
    
    ### ARGUMENTS
    featset = args.featset
    target = args.target
    msi_rsfm = args.msi
    sar_rsfm = args.sar
    n_comp = args.ncomp
    imsize = args.imsize
    device = args.device
    dirpath = args.dirpath

    name_parts = [featset]
    if msi_rsfm!='none':
        name_parts.append(msi_rsfm)
    if sar_rsfm!='none':
        name_parts.append(sar_rsfm)

    sep = '_'
    model_name = sep.join(name_parts)
    model_name = '%s%d'%(model_name,imsize) if imsize!=1000 else model_name
    path_embed = '%s/%d'%(path_embeddings,imsize) if imsize!=1000 else path_embeddings

    path_partition = 'partitions/global/%s_partitions.csv'%target

    ### Load calibration dataset
    calibration_dataset = HabitatDatasetBuilder(path_hab_data=path_hab_data,
                      path_pred_data=path_pred_data,
                      path_partition=path_partition,
                      path_metadata=path_metadata,
                      path_embeddings=path_embed,
                      target=target,featset=featset,plot_id=plot_id,group_id=group_id,class_id=class_id,
                      msi_rsfm=msi_rsfm if msi_rsfm!='none' else None,
                      sar_rsfm=sar_rsfm if sar_rsfm!='none' else None,
                      tabfm=None,n_comp_msi=n_comp,n_comp_sar=n_comp)

    calibration_dataset.build_dataset()

    ### Load config
    with open(config_path,'r') as fp:
        param_dict = json.load(fp)

    param_dict['inputs']['metadata'] = calibration_dataset.feature_metadata
    param_dict['inputs']['categories'] = calibration_dataset.categories

    ### Setting up working directory
    model_dir = '%s/%s/%s'%(work_dir,target,model_name) if dirpath=="none" else dirpath
    os.makedirs(model_dir,exist_ok=True)
    os.makedirs('%s/diagnosis'%model_dir,exist_ok=True)
    os.makedirs('%s/classifiers'%model_dir,exist_ok=True)
    os.makedirs('%s/evaluation'%model_dir,exist_ok=True)
    os.makedirs('%s/explanation'%model_dir,exist_ok=True)
    os.makedirs('%s/forecasting'%model_dir,exist_ok=True)
    os.makedirs('%s/validation'%model_dir,exist_ok=True)

    ### Extract data
    tab_data = calibration_dataset.tab_pred_data.copy()
    msi_data = calibration_dataset.msi_data
    sar_data = calibration_dataset.sar_data

    all_data = [tab_data]
    if msi_data is not None:
        all_data.append(msi_data)
        
    if sar_data is not None:
        all_data.append(sar_data)

    X = pd.concat(all_data,axis=1) 
    Y = calibration_dataset.habitat_data.copy()

    print(X.shape, Y.shape)

    for fold in range(n_splits):
        np.random.seed(seed)

        ### Output files
        perf_file = '%s/evaluation/perfs_%d.csv'%(model_dir,fold)
        conf_file = '%s/evaluation/confusion_%d.csv'%(model_dir,fold)        
        path_to_meta = '%s/metadata.csv'%model_dir
        config_file = '%s/classifiers/config_%d.json'%(model_dir,fold)
        data_att_file = '%s/classifiers/data_atts_%d.json'%(model_dir,fold)
        model_file = '%s/classifiers/model_%d.json'%(model_dir,fold)
        prep_file = '%s/classifiers/preprocessor_%d.joblib'%(model_dir,fold)
        label_file = '%s/classifiers/label_encoder_%d.joblib'%(model_dir,fold)
        imp_file = '%s/explanation/importance_%d.csv'%(model_dir,fold)

        print('Fitting %s model on fold %d, target: %s'%(model_name,fold,target))
        train_idx, test_idx = calibration_dataset.get_split_indices(fold=fold,fold_var='fold_mc')
        
        if os.path.exists(model_file):
            print('Already fitted, loading !')
            habitat_model = XGBoostHDM(model_name='%s_%d'%(model_name,fold), target='all', param_dict=None, k_list=[3,5,10])
            habitat_model.load_config('%s/classifiers/config_%d.json'%(model_dir,fold),
                                      path_to_metadata='%s/metadata.csv'%model_dir)
            habitat_model.load_data_atts('%s/classifiers/data_atts_%d.json'%(model_dir,fold))
            habitat_model.load_preprocessor('%s/classifiers/preprocessor_%d.joblib'%(model_dir,fold))
            habitat_model.load_label_encoder('%s/classifiers/label_encoder_%d.joblib'%(model_dir,fold))
            habitat_model.load_model('%s/classifiers/model_%d.json'%(model_dir,fold))            
        else:
            #### Training
            habitat_model = XGBoostHDM(model_name='%s_%d'%(model_name,fold), target='all', param_dict=param_dict, k_list=[3,5,10])
            habitat_model.fit(X=X.loc[train_idx],y=Y.loc[train_idx,'EUNIS3'],X_val=X.loc[test_idx],y_val=Y.loc[test_idx,'EUNIS3'])
            fig = habitat_model.plot_learning()
            fig.savefig('%s/diagnosis/learning_curve_%d.png'%(model_dir,fold))  
        
            #### Evaluation
            perf, conf_mat = habitat_model.evaluate(X.loc[test_idx],Y.loc[test_idx,'EUNIS3'])
            perf.to_csv(perf_file,index=None)
            conf_mat.to_csv(conf_file,index=None)
        
            #### Export
            habitat_model.param_dict['inputs']['metadata'].to_csv(path_to_meta)
            habitat_model.export_config(config_file,path_to_metadata=path_to_meta)
            habitat_model.export_data_atts(data_att_file)
            habitat_model.save_model(model_file)
            habitat_model.save_preprocessor(prep_file)
            habitat_model.save_label_encoder(label_file)        

        if os.path.exists(imp_file)==False:
            #### Importance
            feat_imp = pd.DataFrame(data=habitat_model.model.feature_importances_,
                                    columns=['importance'],index=habitat_model.model.feature_names_in_)
            feat_imp['feature_code']=habitat_model.original_feats
            feat_imp = pd.merge(feat_imp,calibration_dataset.feature_metadata[['feature_group','feature_code']],how='left')
            feat_imp.to_csv(imp_file)

        if os.path.exists('%s/forecasting/train_xgb_%d.parquet'%(model_dir,fold))==False:
            ### Prediction on train fold
            Yhat_train = habitat_model.predict_proba(X.loc[train_idx])
            Yhat_train.to_parquet('%s/forecasting/train_xgb_%d.parquet'%(model_dir,fold),compression='gzip')

        if os.path.exists('%s/forecasting/valid_xgb_%d.parquet'%(model_dir,fold))==False:
            ### Prediction on test fold
            Yhat_test = habitat_model.predict_proba(X.loc[test_idx])
            Yhat_test.to_parquet('%s/forecasting/valid_xgb_%d.parquet'%(model_dir,fold),compression='gzip')

    print('END ! ')
    
