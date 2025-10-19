from typing import Dict, Tuple, Any
from types import SimpleNamespace
import torch
from torch import nn

from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_group_matrix

class Embedding1dLayer(nn.Module):
    """
    Enables different values in a categorical features to have different embeddings
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Tuple[int, int],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
    ):
        super(Embedding1dLayer, self).__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input

        # Embedding layers
        self.cat_embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in categorical_embedding_dims])
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

    def forward(self, continuous_data: torch.Tensor, categorical_data: torch.Tensor) -> torch.Tensor:
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(categorical_data[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
            # (B, N, C + C)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
                
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
            
        return embed


class CategoryEmbeddingMLP(nn.Module):
    def __init__(self,config):
        super().__init__()

        ### Setting up config
        self.data_config = config['data']
        self.hparams = config['hparams']
        
        ## Setting up dimensions for feature extractors
        self.data_config.embed_dims = [min(50,cat//2) for cat in self.data_config.categorical_cardinalities]
        self.data_config.input_dim = self.data_config.continuous_dim + sum(self.data_config.embed_dims)

        self._build_network()

    def _build_network(self):        
        ## Input processing / embedding layers
        categorical_embedding_dims = [(x,y) for x,y in zip(self.data_config.categorical_cardinalities,self.data_config.embed_dims)]
        self.embedding = Embedding1dLayer(self.data_config.continuous_dim,categorical_embedding_dims,
                                          self.hparams.embedding_dropout,self.hparams.batch_norm_continuous_input)

        ## Hidden layers
        layers = []
        input_dim = self.data_config.input_dim
        hidden_layers = [x for x in self.hparams.hidden_layers.split("-") if x!=""]
        if len(hidden_layers)>0:
            for hidden_dim in hidden_layers:
                hidden_dim = int(hidden_dim)
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                if self.hparams.activation=='leaky_relu':
                    layers.append(nn.LeakyReLU())
                else:
                    layers.append(nn.ReLU())
                    
                layers.append(nn.Dropout(self.hparams.dropout))
                input_dim = hidden_dim  # Update for next layer
            
            self.backbone = nn.Sequential(*layers)
        else:
            self.backbone = nn.Identity()
        
        # Final Output Layer
        self.head = nn.Linear(input_dim, self.data_config.output_dim)

    def forward(self, x_cont, x_cat):
        embed = self.embedding(x_cont,x_cat)
        features = self.backbone(embed)
        logit = self.head(features)
        
        return logit

class TabNetModel(nn.Module):
    def __init__(self, config, feature_groups, device):
        super(TabNetModel, self).__init__()
        
        ### Setting up config
        self.data_config = config['data']
        self.hparams = config['hparams']
        self.feature_groups = feature_groups
        self.device = device

        ## Setting up dimensions for feature extractors
        self.data_config.embed_dims = [min(50,cat//2) for cat in self.data_config.categorical_cardinalities]
        self.data_config.input_dim = self.data_config.continuous_dim + sum(self.data_config.embed_dims)        
        
        self._build_network()        

    def _build_network(self):
        input_dim = self.data_config.continuous_dim+len(self.data_config.categorical_cardinalities)
        group_matrix = create_group_matrix(self.feature_groups,input_dim).to(self.device)
        self.tabnet = TabNet(
            ### Input params
            input_dim=input_dim,
            output_dim=self.data_config.output_dim,
            cat_idxs=[i for i,_ in enumerate(self.data_config.categorical_cardinalities)],
            cat_dims=self.data_config.categorical_cardinalities,
            cat_emb_dim=self.data_config.embed_dims,
            ### Feature grouping
            group_attention_matrix=group_matrix,
            ### Archi params
            n_d=self.hparams.n_d,
            n_a=self.hparams.n_a,
            n_steps=self.hparams.n_steps,
            gamma=self.hparams.gamma,
            n_independent=self.hparams.n_independent,
            n_shared=self.hparams.n_shared,
            mask_type=self.hparams.mask_type,
        )

    def forward(self, x_cont, x_cat):

        self.tabnet.embedder.embedding_group_matrix = self.tabnet.embedder.embedding_group_matrix.to(x_cont.device)
        self.tabnet.tabnet.encoder.group_attention_matrix = self.tabnet.tabnet.encoder.group_attention_matrix.to(x_cont.device)
        
        x_in = torch.cat((x_cat,x_cont),dim=1)
        x, _ = self.tabnet(x_in)
        return x