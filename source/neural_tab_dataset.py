import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class HabitatTabDataset(Dataset):
    def __init__(self, file_path, categorical_cols, numerical_cols, target_col, label_encoder=None):
        """
        Args:
            file_path (str): Path to the Parquet file.
            categorical_cols (list): List of already encoded categorical column names.
            numerical_cols (list): List of numerical column names.
            target_col (str): Target column name (integer class labels for multiclass classification).
        """
        # Load data
        self.num_pred_data = pd.read_parquet(file_path,columns=numerical_cols).values
        self.cat_pred_data = pd.read_parquet(file_path,columns=categorical_cols).values
        self.target_data = pd.read_parquet(file_path,columns=[target_col])

        # Store column names
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col

        # label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder() 
            self.label_encoder.fit(self.target_data[self.target_col])
        else:
            self.label_encoder = label_encoder
            
        self.labels = self.label_encoder.transform(self.target_data[self.target_col])

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        # Convert numerical features to float tensor
        numerical_features = torch.tensor(self.num_pred_data[idx], dtype=torch.float32)

        # Convert categorical features (already encoded) to long tensor
        categorical_features = torch.tensor(self.cat_pred_data[idx], dtype=torch.long)
        
        # Convert target to long tensor for multiclass classification
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return (numerical_features, categorical_features), label

    def compute_class_freq(self):
        class_counts = np.bincount(self.labels)
        return class_counts