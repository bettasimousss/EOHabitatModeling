from copy import deepcopy
import json
from .habitat_model import HabitatModel

class BiogeoHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prefit(self,X,y):
        pass

    def prepare_dataset(self,X,y):
        pass
        
    def fit(self, X,y):
        att_name=self.param_dict['att']
        class_id=self.param_dict['class']

        train_data = X.copy()
        train_data[class_id] = y
        
        self.model = train_data.pivot_table(index=att_name, columns=class_id,fill_value=0,aggfunc=len)
        self.model = self.model.apply(lambda x: x/sum(x),axis=1)
        self.model.loc['OTHER',:] = self.model.sum().values
        
        self.labels = self.model.columns.tolist()
        self.clusters = self.model.index.tolist()
        
    def predict_proba(self, X, output_logit=False):
        pool = self.clusters
        att = self.param_dict['att']
        
        X_in = X[[att]].copy()
        other = X_in.query('%s not in @pool'%att).index
        X_in.loc[other,att] = 'OTHER'
        
        y_hat = self.model.loc[X_in[att].tolist(),:]
        y_hat.index = X_in.index
        
        return y_hat

    def save_model(self,out_file):
        self.model.to_csv(out_file)

    def load_model(self,out_file):
        self.model = pd.read_csv(out_file,index_col=0)
        self.labels = self.model.columns.tolist()
        self.clusters = self.model.index.tolist()

    ### SAVING CONFIG
    def export_config(self,out_file,path_to_metadata=None):
        exported = deepcopy(self.param_dict)
        with open(out_file,'w') as fp:
            json.dump(exported,fp,indent=True)
            
    def load_config(self,out_file,path_to_metadata=None):
        with open(out_file,'r') as fp:
            param_dict = json.load(fp)  
            
        self.param_dict = param_dict