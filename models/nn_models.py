from models.dgcnn import DGCNNFeatureSpace, DGCNNSemanticSegmentor
import torch
import numpy as np
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from models.datasets import SorghumDataset
from models.dgcnn import *
from collections import namedtuple
from models.utils import BNMomentumScheduler, SpaceSimilarityLossV2, LeafMetrics

class SorghumPartNetSemantic(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(SorghumPartNetSemantic,self).__init__()

        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple('args', 'k')
        args = MyStruct(k=15)

        self.DGCNN_semantic_segmentor = DGCNNSemanticSegmentor(args)

        self.save_hyperparameters()
        
    def forward(self, xyz):
        
        # Normalization (min max)
        mins,_ = torch.min(xyz,axis=1)
        maxs,_ = torch.max(xyz,axis=1)
        mins = mins.unsqueeze(1)
        maxs = maxs.unsqueeze(1)
        xyz = (xyz-mins)/(maxs-mins) - 0.5

        # Semantic Label Prediction
        semantic_label_pred = self.DGCNN_semantic_segmentor(xyz)
        
        return semantic_label_pred

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            self.lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = SorghumDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,_,semantic_label,_,_ = batch
        
        pred_semantic_label = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)
        
        with torch.no_grad():
            semantic_label_acc = (torch.argmax(pred_semantic_label, dim=1) == semantic_label).float().mean()
        
        tensorboard_logs = {
            'train_semantic_label_loss': semantic_label_loss, 
            'train_semantic_label_acc': semantic_label_acc
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': semantic_label_loss, 'log': tensorboard_logs}


    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points,_,semantic_label,_,_ = batch

        pred_semantic_label = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        semantic_label_acc = (torch.argmax(pred_semantic_label, dim=1) == semantic_label).float().mean()
        
        tensorboard_logs = {
            'val_semantic_label_loss': semantic_label_loss,
            'val_semantic_label_acc': semantic_label_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        
        semantic_label_loss = torch.stack([x['val_semantic_label_loss'] for x in outputs]).mean()
        semantic_label_acc = torch.stack([x['val_semantic_label_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'val_semantic_label_loss': semantic_label_loss,
            'val_semantic_label_acc': semantic_label_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': semantic_label_loss, 'log': tensorboard_logs}

class SorghumPartNetInstance(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(SorghumPartNetInstance,self).__init__()

        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple('args', 'k')
        args = MyStruct(k=15)

        self.DGCNN_feature_space = DGCNNFeatureSpace(args)

        self.save_hyperparameters()
        
    def forward(self, xyz):
        
        # Normalization (min max)
        mins,_ = torch.min(xyz,axis=1)
        maxs,_ = torch.max(xyz,axis=1)
        mins = mins.unsqueeze(1)
        maxs = maxs.unsqueeze(1)
        xyz = (xyz-mins)/(maxs-mins) - 0.5

        # Instance 
        dgcnn_features = self.DGCNN_feature_space(xyz)

        return dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            self.lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = SorghumDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,_,_,_,leaf = batch
        
        pred_leaf_features = self(points)

        criterion_cluster = SpaceSimilarityLossV2()
        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetrics(self.hparams['leaf_space_threshold'])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features,leaf)
        
        tensorboard_logs = {
            'train_leaf_loss': leaf_loss,
            'train_leaf_accuracy': Acc,
            'train_leaf_precision': Prec,
            'train_leaf_recall': Rec,
            'train_leaf_f1': F
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': leaf_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points,_,_,_,leaf = batch

        pred_leaf_features = self(points)

        criterion_cluster = SpaceSimilarityLossV2()
        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetrics(self.hparams['leaf_space_threshold'])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features,leaf)
        
        tensorboard_logs = {
            'val_leaf_loss': leaf_loss,
            'val_leaf_accuracy': Acc,
            'val_leaf_precision': Prec,
            'val_leaf_recall': Rec,
            'val_leaf_f1': F
            }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_leaf_loss = torch.stack([x['val_leaf_loss'] for x in outputs]).mean()
        Acc = torch.stack([x['val_leaf_accuracy'] for x in outputs]).mean()
        Prec = torch.stack([x['val_leaf_precision'] for x in outputs]).mean()
        Rec = torch.stack([x['val_leaf_recall'] for x in outputs]).mean()
        F = torch.stack([x['val_leaf_f1'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_leaf_loss': val_leaf_loss,
            'val_leaf_accuracy': Acc,
            'val_leaf_precision': Prec,
            'val_leaf_recall': Rec,
            'val_leaf_f1': F
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_leaf_loss, 'log': tensorboard_logs}