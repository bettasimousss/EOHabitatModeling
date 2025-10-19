import json
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics.functional.classification as torchmf

from source.neural_tab_dataset import *
from source.neural_models import *
from source.loss import *

seed = 1234

class NeuralHDM(object):
    def __init__(self,archi,model_name=None, problem=None, param_dict=None, k_list=[3,5,10],*args, **kwargs):
        self.model_name=model_name
        self.problem=problem
        self.archi = archi

        ### Configs
        self.data_params = param_dict['inputs']
        self.hyperparams = param_dict['hyperparams']
        self.train_params = param_dict['train_params']
        
        ### Data parameters
        self.covariates = None
        self.labels=None
        self.num_classes = None
        
        ### Evaluation params
        self.k_list=k_list


    def prepare_dataset(self,train_file,valid_file):
         ### Prepare tabular datasets
        train_dataset = HabitatTabDataset(file_path=train_file, 
                                          categorical_cols=self.data_params['cat_col_names'], 
                                          numerical_cols=self.data_params['num_col_names'],
                                          target_col=self.data_params['target_col'])

        self.covariates = self.data_params['num_col_names'] + self.data_params['cat_col_names']
        self.labels = train_dataset.label_encoder.classes_
        self.num_classes = len(self.labels)
        
        valid_dataset = HabitatTabDataset(file_path=valid_file,
                                          categorical_cols=self.data_params['cat_col_names'], 
                                          numerical_cols=self.data_params['num_col_names'],
                                          target_col=self.data_params['target_col'],
                                          label_encoder=train_dataset.label_encoder)


        return train_dataset, valid_dataset

    def create_archi(self,device='cpu'):
        ### Setup configs
        data_config = {
            'continuous_dim':len(self.data_params['num_col_names']),
            'categorical_cardinalities':[len(cat) for k,cat in self.data_params['categories'].items()],
            'output_dim':self.num_classes
        }

        config = {'data':SimpleNamespace(**data_config),'hparams':SimpleNamespace(**self.hyperparams)}  

        if self.archi=="mlp":
            mlp_model = CategoryEmbeddingMLP(config)
        else:
            use_groups = self.hyperparams['use_groups']

            if use_groups:
                features = self.data_params['cat_col_names'] + self.data_params['num_col_names']
                grp_list = [
                        [features.index(col) for col in grp if col in features] for grp in self.data_params['feature_groups']
                ]
            else:
                grp_list = None
            
            mlp_model = TabNetModel(config,feature_groups=grp_list,device=device)

        return mlp_model

    def fit(self, train_dataset, valid_dataset, log_dir='.', ckpt_path=None):
       
        ### Class imbalance arguments
        cls_num_samples = train_dataset.compute_class_freq()
        total_samples = cls_num_samples.sum()

        ### Define loss
        loss_params = self.train_params['loss']
        criterion = ClassBalancedLoss(cls_num_samples, beta=loss_params['beta'], num_classes=self.num_classes, 
                                  loss_func=FocalLoss(self.num_classes, alpha=loss_params['alpha'], 
                                                      gamma=loss_params['gamma'], reduction="none",
                                                      label_smoothing=loss_params['label_smoothing']))

        ### Create and intialize model
        init_bias = self.train_params['init_bias']
        optim_params = self.train_params['optimizer']

        pl.seed_everything(seed)
        mlp_model = self.create_archi()
        if self.archi=='mlp':
            self.model = MlpHabitatModel(model=mlp_model,num_classes=self.num_classes,
                                         criterion=criterion,lr=optim_params['lr'],decay=optim_params['decay'],optimizer=optim_params['func'])
        else:
            self.model = TabNetHabitatModel(model=mlp_model,num_classes=self.num_classes,
                                            criterion=criterion,lr=optim_params['lr'],decay=optim_params['decay'],optimizer=optim_params['func'])            
        if init_bias:
            log_prior = torch.log(torch.Tensor(cls_num_samples / total_samples))
            self.model._initialize_bias(-log_prior)

        ### Setup dataloaders
        batch_size = self.train_params['batch_size']
        num_workers = self.train_params['num_workers']
        pin_memory = self.train_params['pin_memory']
        max_epochs = self.train_params['max_epochs']
        
        train_dl = DataLoader(train_dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=True,num_workers=num_workers)#,sampler=train_sampler)
        valid_dl = DataLoader(valid_dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=False,num_workers=num_workers)#,sampler=valid_sampler)

        ### Train the model
        logger = pl.loggers.CSVLogger(save_dir=log_dir,name=self.model_name)
        progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=100)  # Update every 10 steps
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        
        trainer = pl.Trainer(accelerator='gpu',callbacks=[progress_bar,lr_monitor],max_epochs=max_epochs,strategy='auto',logger=logger)
        trainer.fit(model=self.model, train_dataloaders=train_dl,val_dataloaders=valid_dl,ckpt_path=ckpt_path) ## set ckpt_path to continue halted training

        ### Save training history
        # Access the log directory
        csv_log_dir = logger.log_dir  # e.g., 'logs/my_model/version_0'
        # Full path to the metrics log file
        metrics_file = f"{csv_log_dir}/metrics.csv"

        df=pd.read_csv(metrics_file)
        train_hist_df = df[['epoch','train_loss','train_recall','train_top3','train_top5','train_top10']].dropna()
        train_hist_df.columns = ['epoch','loss','recall','top3','top5','top10']
        train_hist_df['dataset']='train'
        val_hist_df = df[['epoch','val_loss','val_recall','val_top3','val_top5','val_top10']].dropna()
        val_hist_df.columns = ['epoch','loss','recall','top3','top5','top10']
        val_hist_df['dataset']='valid'

        hist_df = pd.concat([train_hist_df,val_hist_df],axis=0)
        hist_df.to_csv('%s/%s/history.csv'%(log_dir,self.model_name))
    
    def predict_proba(self, pred_dataset,output_logit=False):
        ### Create data loader
        batch_size = self.train_params['batch_size']
        num_workers = self.train_params['num_workers']
        pin_memory = self.train_params['pin_memory']
        pred_dl = DataLoader(pred_dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=False,num_workers=num_workers)

        ### Do model prediction
        self.model.eval()
        print('Using model in inference mode')
        trainer = pl.Trainer(accelerator='gpu',callbacks=[],strategy='auto',logger=None)
        pred_list = trainer.predict(self.model,pred_dl)

        ### Format output
        y_logit = torch.cat(pred_list,dim=0)
        if output_logit:
            y_hat = y_logit
        else:
            y_hat = nn.functional.softmax(y_logit,dim=1)
        
        Y_score = pd.DataFrame(data=y_hat.numpy(),columns=self.labels,index=pred_dataset.target_data.index)

        return Y_score

    def load_model(self,file):
        ### Instantiate model
        mlp_model = self.create_archi()

        if self.archi=="mlp":
            self.model = MlpHabitatModel(model=mlp_model,num_classes=self.num_classes,criterion=None)
        else:
            self.model = TabNetHabitatModel(model=mlp_model,num_classes=self.num_classes,criterion=None)
        
        ### Load weights
        state_dict = torch.load(file,weights_only=True)

        ### Create module
        self.model.load_state_dict(state_dict)
    
    def save_model(self,file):
        torch.save(self.model.state_dict(), file)

    def save_labels(self,file):
        with open(file,'w') as fp:
            json.dump(self.labels,fp)  

    def load_labels(self,file):
        with open(file,'w') as fp:
            self.labels = json.load(fp)  

        self.num_classes = len(self.labels)

class TabNetHabitatModel(pl.LightningModule):
    def __init__(self, model, num_classes, criterion,optimizer='adam',lr=0.01,decay=0.0):
        super().__init__()
        self.criterion = criterion
        self.decay = decay
        
        self.model = model.to(self.device)
        self.lr = lr
        self.optimizer = optimizer
        self.num_classes = num_classes
        
    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (x_cont, x_cat), y = batch
        y_logit = self.model(x_cont,x_cat)
        loss = self.criterion(y_logit, y)

        # Metrics
        preds = torch.argmax(y_logit, dim=1)
        macro_recall = torchmf.multiclass_recall(preds,y,num_classes=self.num_classes,average='macro')
        macro_precision = torchmf.multiclass_precision(preds,y,num_classes=self.num_classes,average='macro')

        # Top-K accuracies
        top3_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=3)
        top5_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=5)
        top10_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=10)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False) 
        self.log("train_recall", macro_recall, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_precision", macro_precision, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        
        self.log("train_top3", top3_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_top5", top5_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_top10", top10_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        
        return loss

    def predict_step(self,batch,batch_idx):
        (x_cont, x_cat), _ = batch
        y_logit = self.model(x_cont,x_cat)
        
        return y_logit

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        (x_cont, x_cat), y = batch
        y_logit = self.model(x_cont,x_cat)

        ## Loss
        val_loss = self.criterion(y_logit, y)

        ## Metrics to track during validation
        preds = torch.argmax(y_logit, dim=1)
        macro_recall = torchmf.multiclass_recall(preds,y,num_classes=self.num_classes,average='macro')
        macro_precision = torchmf.multiclass_precision(preds,y,num_classes=self.num_classes,average='macro')
        
        # Top-K accuracies
        top3_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=3)
        top5_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=5)
        top10_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=10)     

        ## Logging
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_recall", macro_recall, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_precision", macro_precision, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        self.log("val_top3", top3_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_top5", top5_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_top10", top10_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)        

        return val_loss

    def test_step(self, batch, batch_idx):
        # this is the validation loop
        (x_cont, x_cat), y = batch
        y_logit = self.model(x_cont,x_cat)

        ## Metrics to evaluate on test set
        preds = torch.argmax(y_logit, dim=1)
        macro_recall = torchmf.multiclass_recall(preds,y,num_classes=self.num_classes,average='macro')
        macro_precision = torchmf.multiclass_precision(preds,y,num_classes=self.num_classes,average='macro')

        # Top-K accuracies
        top3_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=3)
        top5_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=5)
        top10_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=10)             

        ## Logging
        self.log("test_recall", macro_recall, on_epoch=True, on_step=False)
        self.log("test_precision", macro_precision, on_epoch=True, on_step=False)

        self.log("test_top3", top3_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("test_top5", top5_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("test_top10", top10_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)     

    def configure_optimizers(self):
        if self.optimizer=='SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, nesterov=True, momentum=0.9,weight_decay=self.decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
        
class MlpHabitatModel(pl.LightningModule):
    def __init__(self, model, num_classes, criterion,optimizer='adam',lr=0.01,decay=0.0):
        super().__init__()
        self.criterion = criterion
        self.decay = decay
        
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.num_classes = num_classes

    def _initialize_bias(self,init_bias):
        self.model.head.bias.data = init_bias
        
    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (x_cont, x_cat), y = batch
        y_logit = self.model(x_cont,x_cat)
        loss = self.criterion(y_logit, y)

        # Metrics
        preds = torch.argmax(y_logit, dim=1)
        macro_recall = torchmf.multiclass_recall(preds,y,num_classes=self.num_classes,average='macro')
        macro_precision = torchmf.multiclass_precision(preds,y,num_classes=self.num_classes,average='macro')

        # Top-K accuracies
        top3_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=3)
        top5_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=5)
        top10_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=10)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False) 
        self.log("train_recall", macro_recall, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_precision", macro_precision, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        
        self.log("train_top3", top3_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_top5", top5_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("train_top10", top10_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        
        return loss

    def predict_step(self,batch,batch_idx):
        (x_cont, x_cat), _ = batch
        y_logit = self.model(x_cont,x_cat)
        #y_hat = nn.functional.softmax(y_logit,dim=1)
        
        return y_logit

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        (x_cont, x_cat), y = batch
        y_logit = self.model(x_cont,x_cat)

        ## Loss
        val_loss = self.criterion(y_logit, y)

        ## Metrics to track during validation
        preds = torch.argmax(y_logit, dim=1)
        macro_recall = torchmf.multiclass_recall(preds,y,num_classes=self.num_classes,average='macro')
        macro_precision = torchmf.multiclass_precision(preds,y,num_classes=self.num_classes,average='macro')
        
        # Top-K accuracies
        top3_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=3)
        top5_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=5)
        top10_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=10)     

        ## Logging
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_recall", macro_recall, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_precision", macro_precision, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        self.log("val_top3", top3_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_top5", top5_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_top10", top10_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)        

        return val_loss

    def test_step(self, batch, batch_idx):
        # this is the validation loop
        (x_cont, x_cat), y = batch
        y_logit = self.model(x_cont,x_cat)

        ## Metrics to evaluate on test set
        preds = torch.argmax(y_logit, dim=1)
        macro_recall = torchmf.multiclass_recall(preds,y,num_classes=self.num_classes,average='macro')
        macro_precision = torchmf.multiclass_precision(preds,y,num_classes=self.num_classes,average='macro')

        # Top-K accuracies
        top3_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=3)
        top5_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=5)
        top10_accuracy = torchmf.multiclass_accuracy(y_logit,y,num_classes=self.num_classes, top_k=10)             

        ## Logging
        self.log("test_recall", macro_recall, on_epoch=True, on_step=False)
        self.log("test_precision", macro_precision, on_epoch=True, on_step=False)

        self.log("test_top3", top3_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("test_top5", top5_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("test_top10", top10_accuracy, prog_bar=True, logger=True, on_epoch=True, on_step=False)     

    def configure_optimizers(self):
        if self.optimizer=='SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, 
                                        nesterov=True, momentum=0.9,weight_decay=self.decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
