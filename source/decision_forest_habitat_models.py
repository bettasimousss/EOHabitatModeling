from .habitat_model import HabitatModel
import xgboost as xgb

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostHDM(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_dataset(self, X=None, y=None):
        X_prep, y_prep = super().prepare_dataset(X,y)
        if X_prep is not None:
            if len(self.cat_vars)>0:
                X_prep[self.cat_vars]=X_prep[self.cat_vars].astype('category')

            X_prep[self.num_vars]=X_prep[self.num_vars].astype(float)            
            
        return X_prep, y_prep
        
    def fit(self, X, y, X_val, y_val):
        self.prefit(X, y)
        X_prep, y_prep = self.prepare_dataset(X,y)
        X_val_prep, y_val_prep = self.prepare_dataset(X_val, y_val)
        
        train_params = self.param_dict['train_params']
        es_cbk = xgb.callback.EarlyStopping(rounds=train_params['early_stopping_rounds'], 
                                   metric_name=train_params['eval_metric'], 
                                   data_name='validation_1', 
                                   maximize=train_params['maximize'], 
                                   save_best=True, 
                                   min_delta=train_params['min_delta'])
        
        self.model = xgb.XGBClassifier(callbacks = [es_cbk], **self.param_dict['hyperparams'])
        
        
        if self.param_dict['train_params']['sample_weight']:
            print('using sample weights')
            sweight = self.sample_w
        else:
            sweight = None
            
        self.model.fit(X_prep, y_prep, eval_set=[(X_prep, y_prep), (X_val_prep, y_val_prep)], sample_weight=sweight)        
    
    def predict_proba(self, X,output_logit=False):
        X_prep, _ = self.prepare_dataset(X)
        X_data = xgb.DMatrix(X_prep,enable_categorical=True)
        
        booster = self.model.get_booster()
        Y_hat = pd.DataFrame(data=booster.predict(X_data,output_margin=output_logit),columns=self.labels,index=X.index)
        
        return Y_hat
    
    def plot_learning(self, met_name='mlogloss'):
        history = self.model.evals_result_
        epochs = len(history['validation_0'][met_name])
        
        fig, ax = plt.subplots(1,1)
        sns.lineplot(ax=ax,x=np.arange(epochs),y=history['validation_0'][met_name], label='train',color='blue')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=history['validation_0'][met_name], color='blue')

        sns.lineplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation_1'][met_name]), label='valid', color='red')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation_1'][met_name]), color='red') 
        ax.set_xlabel('Boosting rounds')
        ax.set_ylabel(met_name)   
        
        return fig  

    def load_model(self,file):
        self.model = xgb.XGBClassifier()
        self.model.load_model(file)
    
    def save_model(self,file):
        self.model.save_model(file)    


class RandomForestHDM(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_dataset(self, X=None, y=None):
        X_prep, y_prep = super().prepare_dataset(X,y)
        if X_prep is not None:
            if len(self.cat_vars)>0:
                X_prep[self.cat_vars]=X_prep[self.cat_vars].astype('category')

            X_prep[self.num_vars]=X_prep[self.num_vars].astype(float)            
            
        return X_prep, y_prep
        
    def fit(self, X, y):
        self.prefit(X, y)
        X_prep, y_prep = self.prepare_dataset(X,y)
        
        train_params = self.param_dict['train_params']
        self.model = xgb.XGBRFClassifier(**self.param_dict['hyperparams'])
        
        
        if self.param_dict['train_params']['sample_weight']:
            print('using sample weights')
            sweight = self.sample_w
        else:
            sweight = None
            
        self.model.fit(X_prep, y_prep, sample_weight=sweight)        
    
    def predict_proba(self, X,output_logit=False):
        X_prep, _ = self.prepare_dataset(X)
        X_data = xgb.DMatrix(X_prep,enable_categorical=True)
        
        booster = self.model.get_booster()
        Y_hat = pd.DataFrame(data=booster.predict(X_data,output_margin=output_logit),columns=self.labels,index=X.index)
        
        return Y_hat

    def load_model(self,file):
        self.model = xgb.XGBRFClassifier()
        self.model.load_model(file)
    
    def save_model(self,file):
        self.model.save_model(file)            