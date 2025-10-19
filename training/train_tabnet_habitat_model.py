import os
import joblib
import json

import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from source.evaluation import *
from source.neural_habitat_models import *

with open('dataset/global/data_atts.json','r') as fp:
    num_col_names,cat_col_names = json.load(fp)

with open('dataset/global/categories.json','r') as fp:
    categories = json.load(fp)

with open('dataset/global/feature_groups.json','r') as fp:
    feature_groups = json.load(fp)

with open('configs/best_tabnet.json','r') as fp:
    param_dict = json.load(fp)

param_dict['inputs'] = {
    'num_col_names':num_col_names,
    'cat_col_names':cat_col_names,
    'categories': categories,
    'feature_groups': feature_groups,
    'target_col': 'EUNIS3'
}

data_folder = 'dataset/global'
model_folder = 'habitat_models/global/full/tabnet/'
log_dir = '%s/diagnosis/'%model_folder

n_splits = 5
seed = 1234

if __name__ == "__main__":    
    print('Start of model training')

    os.makedirs('%s/classifiers/'%model_folder,exist_ok=True)
    os.makedirs('%s/diagnosis/'%model_folder,exist_ok=True)
    os.makedirs('%s/evaluation/'%model_folder,exist_ok=True)
    os.makedirs('%s/forecasting/'%model_folder,exist_ok=True)

    parser = argparse.ArgumentParser("Train global habitat model using TabNet", add_help=True)
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold",
    )

    args = parser.parse_args()
    fold = args.fold

    if (fold>n_splits)or(fold<0):
        raise('Inexistent fold, please choose a value between 0 and %d'%n_splits)    
    
    print('Fold %d'%fold)
    
    model_name = 'tabnet_%d'%fold
    
    train_file = '%s/train_%d.parquet'%(data_folder,fold)
    valid_file = '%s/valid_%d.parquet'%(data_folder,fold)

    if os.path.exists('%s/classifiers/%s.pth'%(model_folder,model_name)):
        print('Already trained, loading weights !')
        mlp_hdm = NeuralHDM(archi='tabnet',model_name=model_name, problem='all', param_dict=param_dict, k_list=[3,5,10])
        train_dataset, valid_dataset = mlp_hdm.prepare_dataset(train_file,valid_file) ## this is needed to set labels and forecasting
        mlp_hdm.load_labels('%s/classifiers/labels_%d.json'%(model_folder,fold))
        mlp_hdm.load_model('%s/classifiers/%s.pth'%(model_folder,model_name))
    else:
        ### Training
        print('Training')
        mlp_hdm = NeuralHDM(archi='tabnet',model_name=model_name, problem='all', param_dict=param_dict, k_list=[3,5,10])
        train_dataset, valid_dataset = mlp_hdm.prepare_dataset(train_file,valid_file)
        mlp_hdm.fit(train_dataset, valid_dataset, log_dir=log_dir, ckpt_path=None)

        ### Saving model
        print('Saving model')
        mlp_hdm.save_model('%s/classifiers/%s.pth'%(model_folder,model_name))
        mlp_hdm.save_labels('%s/classifiers/labels_%d.json'%(model_folder,fold))

        ### Diagnosis curves
        hist_df = pd.read_csv('%s/%s/history.csv'%(log_dir,model_name),index_col=0)
        fig, ax = plt.subplots(1,1)
        sns.lineplot(ax=ax,data=hist_df,x='epoch',y='loss',hue='dataset')
        fig.suptitle('Learning curve %s'%model_name)
        fig.savefig('%s/loss_%s.png'%(log_dir,model_name))

    ### Prediction
    print('Forecasting')
    if os.path.exists('%s/forecasting/train_%s.parquet'%(model_folder,model_name))==False:
        Y_train_score = mlp_hdm.predict_proba(train_dataset,output_logit=False)
        Y_train_score.to_parquet('%s/forecasting/train_%s.parquet'%(model_folder,model_name),compression='gzip')

    if os.path.exists('%s/forecasting/valid_%s.parquet'%(model_folder,model_name))==False:
        Y_val_score = mlp_hdm.predict_proba(valid_dataset,output_logit=False)
        Y_val_score.to_parquet('%s/forecasting/valid_%s.parquet'%(model_folder,model_name),compression='gzip')
    else:
        print('Already forecasted, loading for evaluation !')
        Y_val_score = pd.read_parquet('%s/forecasting/valid_%s.parquet'%(model_folder,model_name))

    ### Evaluation
    print('Evaluation')
    if os.path.exists('%s/evaluation/conf_mat_%d.csv'%(model_folder,fold))==False:
        perfs, conf_mat = eval_classifier(Y_val_score,valid_dataset.target_data,model_name=model_name,super_class='EUNIS',classes=mlp_hdm.labels)
        perfs.to_csv('%s/evaluation/perfs_%d.csv'%(model_folder,fold))
        conf_mat.to_csv('%s/evaluation/conf_mat_%d.csv'%(model_folder,fold))
    else:
        print('Already evaluated !')


    print('END OF TRAINING !')
