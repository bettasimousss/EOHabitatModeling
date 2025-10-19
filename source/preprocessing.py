from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, QuantileTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Define custom transformer
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].values

# Define custom transformer
class ColumnScaler(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X/self.scale_factor 

# Define custom transformer
class CyclicalTransformer(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, cycle):
        self.cycle = cycle
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        arr = X*(2.*np.pi/self.cycle)
        cos = np.cos(arr)
        sin = np.sin(arr)

        return np.concatenate([cos,sin],axis=1)


class Pandarizer(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, colnames):
        self.colnames = colnames
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(data=X,columns=self.colnames)

        return df   


def create_prep_pipeline(feature_metadata, categories=None, std=False, onehot=1):
    transformer_list = []
    colnames_list = []
    original_names_list = []
    preproc_dict = {}
    for fname, ftype, scale_factor, cycle in feature_metadata[['feature_code','feature_type','scale','cycle']].values:

        if ftype=="binary":
            prep = Pipeline([
                    ('selector', ColumnSelector([fname]))
            ])

            colnames = [fname]

        if ftype=="continuous":
            if std==True:
                prep = Pipeline([
                        ('selector', ColumnSelector([fname])),
                        ('scale', StandardScaler())
                ])

            else:
                prep = Pipeline([
                        ('selector', ColumnSelector([fname]))
                ]) 

            colnames = [fname]

        if ftype=="cyclical":
            prep = Pipeline([
                    ('selector', ColumnSelector([fname])),
                    ('scale', CyclicalTransformer(cycle=cycle))
            ])

            colnames = ['%s_cos'%fname,'%s_sin'%fname]

        if ftype in ["frequency","fraction","daycount"]:
            if std==True:
                prep = Pipeline([
                        ('selector', ColumnSelector([fname])),
                        ('scale', ColumnScaler(scale_factor=scale_factor))
                ])

            else:
                prep = Pipeline([
                        ('selector', ColumnSelector([fname]))
                ])

            colnames = [fname]

        if ftype=="categorical":
            if onehot==1:
                prep = Pipeline([
                    ('selector', ColumnSelector([fname])),
                    ('scale',OneHotEncoder(categories=[categories[fname]], sparse_output=False,handle_unknown='ignore'))
                ])

                colnames=['%s_%s'%(fname,c) for c in categories[fname]]

            elif onehot==2:
                prep = Pipeline([
                    ('selector', ColumnSelector([fname])),
                    ('scale',OrdinalEncoder(categories=[categories[fname]],
                                            dtype=np.float32,handle_unknown='use_encoded_value',unknown_value=np.nan))
                ])

                colnames=[fname]

            else:
                prep = Pipeline([
                    ('selector', ColumnSelector([fname]))
                ])

                colnames=[fname]

        original_names = [fname]*len(colnames)

        ### Add to rest
        transformer_list.append((fname,prep))
        original_names_list += original_names
        colnames_list += colnames

        if len(prep.steps)>1:
            preproc_dict[fname] = (prep.steps[1][1],colnames)


    feature_union = FeatureUnion(transformer_list=transformer_list)
    preprocessor = Pipeline(steps=[
                    ('union', feature_union),
                    ("pandarizer",Pandarizer(colnames = colnames_list))
            ])
    
    return preprocessor, preproc_dict, colnames_list, original_names_list
