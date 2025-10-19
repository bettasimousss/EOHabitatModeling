import pandas as pd
import json
import numpy as np

class HabitatDatasetBuilder(object):
    ### EUNIS formations
    eunis_formations = ['MA2','N','P','Q','R','S','T','U','V']

    ### Feature groups
    env_groups = ['climate','terrain','geology','soil']
    rs_groups = ['soil_moisture','hydrology','phenology','vegetation','landscape']

    ### RSFM list
    msi_rsfm_list = ['dofa_s2','prithvi','ssl4eo_vit_dino','ssl4eo_vit_moco',
                      'seco','eo4b','ssl4eo_cnn_dino','ssl4eo_cnn_moco']

    sar_rsfm_list = ['dofa_s1']    
    
    ### Feature metadata
    def __init__(self,path_hab_data,path_pred_data,path_partition,path_metadata,path_embeddings,
                 target,featset,plot_id='PlotID',group_id='EUNIS1',class_id='EUNIS3',
                 msi_rsfm=None,sar_rsfm=None,tabfm=None,n_comp_msi=100,n_comp_sar=100):

        ### ID variables
        self.plot_id = plot_id
        self.class_id = class_id
        self.group_id = group_id
        
        ### Store paths
        self.path_hab_data = path_hab_data
        self.path_pred_data = path_pred_data
        self.path_partition = path_partition
        self.path_embeddings = path_embeddings
        
        ### Set metadata
        self.feature_metadata = pd.read_csv(path_metadata,index_col=None)
        self.target = target
        self.featset = featset

        ### Set foundation model setting
        self.msi_rsfm = msi_rsfm
        self.n_comp_msi = n_comp_msi
        self.sar_rsfm = sar_rsfm
        self.n_comp_sar = n_comp_sar
        self.tabfm = tabfm

    def build_dataset(self):
        ### Setup covariates
        self.set_tab_covars_list()

        ### Setup msi variables
        self.set_msi_covars_list()

        ### Setup sar variables
        self.set_sar_covars_list()

        ### Get habitat data
        self.habitat_data = self.get_habitat_data()
        self.nb_classes = len(self.habitat_data[self.class_id].unique())

        ### Get tabular predictors
        self.tab_pred_data = self.get_covars_data()

        ### Get MSI / SAR embeddings
        self.msi_data, self.sar_data = self.get_embeddings_data()

        ### Updating metadata file
        tab_covars = self.tab_covars
        feature_metadata = self.feature_metadata.query('feature_code in @tab_covars').copy()
        deep_features = []
        for fcode in self.msi_covars:
            deep_features.append({'feature_group':'MSI', 'feature_code':fcode, 
                                  'feature_type':'continuous', 'nomenclature':np.nan,
                                  'scale':np.nan, 'cycle':np.nan})
            
        for fcode in self.sar_covars:
            deep_features.append({'feature_group':'SAR', 'feature_code':fcode, 
                      'feature_type':'continuous', 'nomenclature':np.nan,
                      'scale':np.nan, 'cycle':np.nan})

        deep_features = pd.DataFrame.from_dict(deep_features)
        self.feature_metadata = pd.concat([feature_metadata,deep_features],axis=0,ignore_index=True)

        ### Categories
        self.categories = dict() 
        for feature_code, nomenclature_path in self.feature_metadata.query('feature_type=="categorical"')[['feature_code','nomenclature']].values:
            with open(nomenclature_path,'r') as fp:
                categs = json.load(fp)
        
            self.categories[feature_code] = [k for k in categs.values()]    
        
    def set_tab_covars_list(self):
        ### Select the list of predictors
        env_groups = self.env_groups
        rs_groups = self.rs_groups

        if self.featset=="env":
            self.tab_covars = self.feature_metadata.query('feature_group in @env_groups')['feature_code'].tolist()
        elif self.featset=="rs":
            self.tab_covars = self.feature_metadata.query('feature_group in @rs_groups')['feature_code'].tolist()
        else:
            self.tab_covars = self.feature_metadata['feature_code'].tolist()

        var_list = self.tab_covars
        self.feature_metadata = self.feature_metadata.query('feature_code in @var_list')

    def set_msi_covars_list(self):
        if self.msi_rsfm:
            self.msi_covars = ['%s_pc%d'%(self.msi_rsfm,c) for c in range(self.n_comp_msi)]
        else:
            self.msi_covars = []
        
    def set_sar_covars_list(self):
        if self.sar_rsfm:
            self.sar_covars = ['%s_pc%d'%(self.sar_rsfm,c) for c in range(self.n_comp_sar)]
        else:
            self.sar_covars = []

    def get_covars_data(self):
        pred_data = pd.read_parquet(self.path_pred_data,columns=self.tab_covars)
        cat_vars = self.feature_metadata.query('feature_type=="categorical"')['feature_code'].tolist()

        for c in cat_vars:
            if c in pred_data.columns:
                pred_data[c]=pred_data[c].astype('category')
        
        return pred_data

    def get_habitat_data(self):
        hab_cols = [self.plot_id,self.group_id,self.class_id]
        habitat_data = pd.read_parquet(self.path_hab_data,columns=hab_cols).set_index(self.plot_id)
        if self.target in self.eunis_formations:
            habitat_data = habitat_data.query('%s=="%s"'%(self.group_id,self.target))

        return habitat_data
            
    def get_embeddings_data(self):
        ### Select the foundation model and get its embeddings
        if self.msi_rsfm:
            msi_data = pd.read_parquet('%s/%s/%s_pc_embedding.parquet'%(self.path_embeddings,self.msi_rsfm,self.msi_rsfm),
                                       columns=self.msi_covars)
        else:
            msi_data = None
            
        if self.sar_rsfm:
            sar_data = pd.read_parquet('%s/%s/%s_pc_embedding.parquet'%(self.path_embeddings,self.sar_rsfm,self.sar_rsfm),
                                       columns=self.sar_covars)
        else:
            sar_data = None
            
        return msi_data, sar_data
        
    def get_split_indices(self,fold,fold_var='fold'):
        data_split = pd.read_csv(self.path_partition,index_col=0)
        train_idx = data_split.query('%s!=%d'%(fold_var,fold)).index
        test_idx = data_split.query('%s==%d'%(fold_var,fold)).index

        return train_idx, test_idx