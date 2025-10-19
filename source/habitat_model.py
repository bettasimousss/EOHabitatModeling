import joblib
import json
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from .preprocessing import *
from .evaluation import *

def compute_cb_weights(labels, beta=0.9999):
    samples_per_cls = np.bincount(labels)
    no_of_classes = len(samples_per_cls)
    
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / weights.sum() * no_of_classes
    
    return weights
    
class HabitatModel(object):
    def __init__(self, model_name=None, problem=None, param_dict=None, k_list=[3,5,10],*args, **kwargs):
        self.model_name=model_name
        self.problem=problem
        self.param_dict=param_dict

        ### preprocessors
        self.preprocessor = None
        self.label_encoder = None
        
        ### problem setting
        self.labels=None
        self.covariates = None
        self.in_covars = None

        ### Evaluation params
        self.k_list=k_list

    def prefit(self,X,y):
        self.preprocessor, self.preproc_dict, self.in_covars, self.original_feats = create_prep_pipeline(self.param_dict['inputs']['metadata'], categories=self.param_dict['inputs']['categories'], std=self.param_dict['inputs']['std'], onehot=self.param_dict['inputs']['onehot'])
        self.label_encoder = LabelEncoder()

        self.preprocessor.fit(X)
        self.label_encoder.fit(y)
        
        self.covariates = X.columns.tolist()
        self.labels = self.label_encoder.classes_

        ## Compute class weights
        y_prep = self.label_encoder.transform(y)
        self.class_w = compute_cb_weights(y_prep,beta=0.9999)
        self.sample_w = self.class_w[y_prep]
        #self.class_w = compute_class_weight(y=y,classes=self.labels,class_weight='balanced')
        #self.sample_w = compute_sample_weight(y=y,class_weight='balanced')

        if self.param_dict['inputs']['onehot']!=1:
            self.cat_vars = self.param_dict['inputs']['metadata'].query('feature_type=="categorical"')['feature_code'].tolist()
        else:
            self.cat_vars = []
                                               
        self.num_vars = list(set(self.in_covars).difference(self.cat_vars))   
        
    def prepare_dataset(self, X=None, y=None):
        X_prep = y_prep = None

        if X is not None:
            X_prep = self.preprocessor.transform(X)

        if y is not None:
            y_prep = self.label_encoder.transform(y)

        return X_prep, y_prep

    def fit(self, X, y):
        pass
        
    def predict_proba(self, X, output_logit=False):
        pass        
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        
        return y_hat.idxmax(axis=1)

    def calibrate(self,X,y,X_val,y_val):
        ### Create calibration model
        self.calibrator = TempScaleCalibration()

        ### Get predicted margin scores
        train_logits = self.model.predict_proba(X, output_logit=True)
        val_logits = self.model.predict_proba(X, output_logit=True)

        ### Fit temperature scaling
        self.calibrator.fit(train_logits,y)

        ### Predict calibrated probabilities
        yhat_train = self.calibrator.predict_proba(train_logits)
        yhat_val = self.calibrator.predict_proba(val_logits)
        
        ### Evaluate on validation dataset
        ece_error = expected_calibration_error(y_val,yhat,n_bins=10)
    
        ### Plot calibration plot
        fig1 = plot_calibration_curve(y, yhat, title="Train calibration plot")
        fig2 = plot_calibration_curve(y_val, yhat, title="Validation calibration plot")

        return fig1, fig2, ece_error
        
    
    def evaluate(self,X,y):
        y_score = self.predict_proba(X)
        perfs, conf_mat = eval_classifier(y_score=y_score,y_true=y, k_list=self.k_list, 
                                          model_name=self.model_name, 
                                          super_class=self.problem, classes=self.labels)
        
        return perfs, conf_mat

    def explain_prediction(self,X):
        pass

    #### SAVING DATA ATTRIBUTES ###
    def export_data_atts(self,out_file):
            misc_params = {
                ### Input stats
                'covariates':self.covariates,
                'original_feats':self.original_feats,
                'in_covars':self.in_covars,
                'num_vars':self.num_vars,
                'cat_vars':self.cat_vars,
    
                ### Dataset stats
                'class_w':self.class_w.tolist()
            }
            
            with open(out_file,'w') as fp:
                json.dump(misc_params, fp,indent=True)

    def load_data_atts(self,out_file):
        with open(out_file,'r') as fp:
            misc_params = json.load(fp)
        
        ### Input stats
        self.covariates = misc_params['covariates']
        self.original_feats = misc_params['original_feats']
        self.in_covars = misc_params['in_covars']
        self.num_vars = misc_params['num_vars']
        self.cat_vars = misc_params['cat_vars']
        
        ### Dataset stats
        self.class_w = misc_params['class_w']

    ### SAVING CONFIG
    def export_config(self,out_file,path_to_metadata):
        exported = deepcopy(self.param_dict)
        exported['inputs']['metadata'] = path_to_metadata
        with open(out_file,'w') as fp:
            json.dump(exported,fp,indent=True)
            
    def load_config(self,out_file,path_to_metadata):
        with open(out_file,'r') as fp:
            param_dict = json.load(fp)  
            
        param_dict['inputs']['metadata'] = pd.read_csv(path_to_metadata,index_col=0)
        self.param_dict = param_dict

    #### SAVING MODEL OBJECTS ####
    def save_preprocessor(self,file):
        joblib.dump(self.preprocessor, file)

    def save_label_encoder(self,file):
        joblib.dump(self.label_encoder, file)    
        
    def load_preprocessor(self,file):
        self.preprocessor = joblib.load(file)

    def load_label_encoder(self,file):
        self.label_encoder = joblib.load(file)
        self.labels = self.label_encoder.classes_
        
    def save_model(self):
        pass
        
    def load_model(self):
        pass


class EnsembleHabitatModel(object):
    def __init__(self, model_names=[], models=[], k_list=[3,5,10], ensemble_name='ensemble', problem=None):
        self.model_names = model_names
        self.models = models
        self.labels = np.unique([c for mod in self.models for c in mod.labels]).tolist()
        
        self.ensemble_name = ensemble_name
        self.problem = problem
        self.k_list = k_list
        
    def predict_proba(self,X):
        n = X.shape[0]
        m = len(self.labels)
        p = len(self.models)
        
        Y_score = pd.DataFrame(data=np.zeros((n,m)),columns=self.labels,index=X.index)
        Y_committee = pd.DataFrame(data=np.zeros((n,m)),columns=self.labels,index=X.index)
        Y_raw = {}
        
        for mod_name, mod in zip(self.model_names, self.models):
            print('Predicting using %s'%mod_name)
            
            ## Sum probas
            y_hat = mod.predict_proba(X)
            Y_score[y_hat.columns]+=y_hat.values
            
            ## Sum indicators
            y_class = pd.get_dummies(y_hat.idxmax(axis=1))
            Y_committee[y_class.columns]+=y_class.values
            
            #Y_raw[mod_name]=y_hat
        
        print('Voting')
        ### Soft voting
        Y_score = Y_score / p
        
        ###  Hard voting
        Y_committee = Y_committee / p
        
        return Y_raw, Y_score, Y_committee
    
    
    def evaluate(self,X,y):
        _, Y_score, Y_committee = self.predict_proba(X)
        
        soft_perfs, soft_conf_mat = eval_classifier(y_score=Y_score,y_true=y, k_list=self.k_list,
                                                    model_name='soft_%s'%self.ensemble_name, super_class=self.problem, classes=self.labels)
        
        hard_perfs, hard_conf_mat = eval_classifier(y_score=Y_committee,y_true=y, 
                                                    k_list=self.k_list,model_name='hard_%s'%self.ensemble_name, 
                                                    super_class=self.problem, classes=self.labels)
        
        return soft_perfs, hard_perfs, soft_conf_mat, hard_conf_mat    