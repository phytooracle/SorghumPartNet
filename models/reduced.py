import torch
import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.datasets import (
    SorghumDataset,
    SorghumDatasetWithNormals,
    TreePartNetDataset,
    PartNetDataset,
)
from collections import namedtuple
from models.utils import (
    BNMomentumScheduler,
    SpaceSimilarityLossV2,
    SpaceSimilarityLossV3,
    SpaceSimilarityLossV4,
    SpaceSimilarityLossV5,
    LeafMetrics,
    SemanticMetrics,
    LeafMetricsTraining,
)

from data.load_raw_data import load_real_ply_with_labels
import matplotlib.pyplot as plt
import torchvision
from sklearn.cluster import DBSCAN
from data.utils import distinct_colors
from models.modules import KNNSpaceRegularizer


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(
        x,
        k=20,
        idx=None,
        dim9=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNNFeatureSpace(nn.Module):
    def __init__(self, args, input_dim=3, device=None):
        super(DGCNNFeatureSpace, self).__init__()
        self._device = device
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = get_graph_feature(
            x, k=self.k, device=self._device
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x1, k=self.k, device=self._device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x2, k=self.k, device=self._device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(
            x3, k=self.k, device=self._device
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        return x4.transpose(1, 2)


class DGCNNSemanticSegmentor(nn.Module):
    def __init__(self, k, output_dim=3, input_dim=3, device=None):
        super(DGCNNSemanticSegmentor, self).__init__()
        self._device = device
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(output_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, output_dim, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = get_graph_feature(
            x, k=self.k, device=self._device
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x1, k=self.k, device=self._device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x2, k=self.k, device=self._device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(
            x3, k=self.k, device=self._device
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        return x4


class SorghumPartNetSemantic(pl.LightningModule):
    def __init__(self, hparams, debug=False, device=None):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(SorghumPartNetSemantic, self).__init__()

        self._device = device
        self.is_debug = debug
        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        self.DGCNN_semantic_segmentor = DGCNNSemanticSegmentor(
            self.hparams["dgcnn_k"],
            input_dim=(
                3 if "input_dim" not in self.hparams else self.hparams["input_dim"]
            ),
            device=self._device
        ).double()

        self.save_hyperparameters()

    def forward(self, xyz):

        # Normalization
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "min-max"
        ):
            mins, _ = torch.min(xyz, axis=1)
            maxs, _ = torch.max(xyz, axis=1)
            mins = mins.unsqueeze(1)
            maxs = maxs.unsqueeze(1)
            xyz = (xyz - mins) / (maxs - mins) - 0.5
        elif self.hparams["normalization"] == "mean-std":
            mean = torch.mean(xyz, axis=1)
            mean = mean.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            std = torch.std(xyz, axis=1)
            std = std.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            xyz = (xyz - mean) / std

        # Semantic Label Prediction
        semantic_label_pred = self.DGCNN_semantic_segmentor(xyz)

        return semantic_label_pred

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        if "use_normals" not in self.hparams:
            dataset = SorghumDataset(ds_path)
        else:
            dataset = SorghumDatasetWithNormals(
                ds_path,
                self.hparams["use_normals"],
                self.hparams["std_noise"],
                self.hparams["duplicate_ground_prob"],
                self.hparams["focal_only_prob"],
                debug=self.is_debug,
            )

        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, semantic_label, _, _ = batch
        else:
            points, semantic_label = batch

        pred_semantic_label = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        with torch.no_grad():
            # semantic_label_acc = (
            #     (torch.argmax(pred_semantic_label, dim=1) == semantic_label)
            #     .float()
            #     .mean()
            # )
            metric_calculator = SemanticMetrics()
            semantic_label_acc = metric_calculator(
                torch.argmax(pred_semantic_label, dim=1), semantic_label
            )

        tensorboard_logs = {
            "train_semantic_label_loss": semantic_label_loss,
            "train_semantic_label_acc": semantic_label_acc,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": semantic_label_loss, "log": tensorboard_logs}

    # def log_pointcloud_image(self,)
    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, semantic_label, _, _ = batch
        else:
            points, semantic_label = batch

        pred_semantic_label = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        # semantic_label_acc = (
        #     (torch.argmax(pred_semantic_label, dim=1) == semantic_label).float().mean()
        # )
        metric_calculator = SemanticMetrics()
        semantic_label_acc = metric_calculator(
            torch.argmax(pred_semantic_label, dim=1), semantic_label
        )

        tensorboard_logs = {
            "val_semantic_label_loss": semantic_label_loss,
            "val_semantic_label_acc": semantic_label_acc,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return tensorboard_logs

    def validation_epoch_end(self, batch):
        self.validation_real_data()

    def validation_real_data(self):
        real_data_path = self.hparams["real_data"]

        semantic_model = self.to(self._device)
        semantic_model.DGCNN_semantic_segmentor.device = self._device

        files = os.listdir(real_data_path)
        accs = []
        pred_images = []

        for file in files:
            path = os.path.join(real_data_path, file)
            points, _, semantic_labels = load_real_ply_with_labels(path)
            points = torch.tensor(points, dtype=torch.float64).to(self._device)
            if (
                "use_normals" in semantic_model.hparams
                and semantic_model.hparams["use_normals"]
            ):
                pred_semantic_label = semantic_model(
                    torch.unsqueeze(points, dim=0).to(self._device)
                )
            else:
                pred_semantic_label = semantic_model(
                    torch.unsqueeze(points[:, :3], dim=0).to(self._device)
                )

            pred_semantic_label = F.softmax(pred_semantic_label, dim=1)
            pred_semantic_label = pred_semantic_label[0].cpu().detach().numpy().T
            pred_semantic_label_labels = np.argmax(pred_semantic_label, 1)

            colors = np.column_stack(
                (
                    pred_semantic_label_labels,
                    pred_semantic_label_labels,
                    pred_semantic_label_labels,
                )
            ).astype("float32")
            colors[colors[:, 0] == 0, :] = [0.3, 0.1, 0]
            colors[colors[:, 0] == 1, :] = [0, 0.7, 0]
            colors[colors[:, 0] == 2, :] = [0, 0, 0.7]

            metric_calculator = SemanticMetrics()
            acc = metric_calculator(
                torch.tensor(pred_semantic_label_labels), torch.tensor(semantic_labels)
            )

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Accuracy: {acc:.2f}")
            fig.canvas.draw()
            X = (
                torch.tensor(np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3])
                .transpose(0, 2)
                .transpose(1, 2)
            )
            X = torchvision.transforms.functional.resize(X, (1000, 1000))
            plt.close(fig)
            accs.append(acc)
            pred_images.append(X)

        accs = torch.tensor(accs)
        grid = torch.cat(pred_images, 1)
        self.logger.experiment.add_image(
            "pred_real_data", grid, self.trainer.current_epoch
        )

        # self.log(
        #     "test_real_acc",
        #     torch.mean(accs),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        self.logger.experiment.add_scalar(
            "test_real_acc", torch.mean(accs), self.trainer.current_epoch
        )


class SorghumPartNetInstance(pl.LightningModule):
    def __init__(self, hparams, debug=False, device=None):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(SorghumPartNetInstance, self).__init__()

        self._device = device
        self.is_debug = debug
        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple("args", "k")
        if "dgcnn_k" in self.hparams:
            args = MyStruct(k=self.hparams["dgcnn_k"])
        else:
            args = MyStruct(k=15)

        self.DGCNN_feature_space = DGCNNFeatureSpace(
            args,
            (3 if "input_dim" not in self.hparams else self.hparams["input_dim"]),
            device=self._device
        ).double()

        if "loss_fn" in self.hparams and self.hparams["loss_fn"] == "knn_space_mean":
            self.space_reqularizer_module = KNNSpaceRegularizer(
                self.hparams["loss_fn_param"]
            )
        else:
            self.space_reqularizer_module = None

        self.save_hyperparameters()

    def forward(self, xyz):

        # Normalization
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "min-max"
        ):
            mins, _ = torch.min(xyz, axis=1)
            maxs, _ = torch.max(xyz, axis=1)
            mins = mins.unsqueeze(1)
            maxs = maxs.unsqueeze(1)
            xyz = (xyz - mins) / (maxs - mins) - 0.5
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "mean-std"
        ):
            mean = torch.mean(xyz, axis=1)
            mean = mean.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            std = torch.std(xyz, axis=1)
            std = std.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            xyz = (xyz - mean) / std

        # Instance
        dgcnn_features = self.DGCNN_feature_space(xyz)

        # Take mean of the k nearest neighbors
        if self.space_reqularizer_module is not None:
            dgcnn_features = self.space_reqularizer_module(xyz, dgcnn_features)

        return dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        if "use_normals" not in self.hparams:
            dataset = SorghumDataset(ds_path)
        else:
            if "dataset" not in self.hparams or self.hparams["dataset"] == "SPNS":
                dataset = SorghumDatasetWithNormals(
                    ds_path,
                    self.hparams["use_normals"],
                    self.hparams["std_noise"],
                    debug=self.is_debug,
                )
            elif self.hparams["dataset"] == "TPN":
                dataset = TreePartNetDataset(
                    ds_path,
                    debug=self.is_debug,
                )
            elif self.hparams["dataset"] == "PN":
                dataset = PartNetDataset(
                    ds_path,
                    debug=self.is_debug,
                )

        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, _, _, leaf = batch
        else:
            points, leaf = batch

        pred_leaf_features = self(points)

        if "loss_fn" not in self.hparams or self.hparams["loss_fn"] == "v2":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v3":
            criterion_cluster = SpaceSimilarityLossV3(points)
        elif self.hparams["loss_fn"] == "v4":
            criterion_cluster = SpaceSimilarityLossV4(points)
        elif self.hparams["loss_fn"] == "knn_space_mean":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v5":
            criterion_cluster = SpaceSimilarityLossV5(points)

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetricsTraining(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "train_leaf_loss": leaf_loss,
            "train_leaf_accuracy": Acc,
            "train_leaf_precision": Prec,
            "train_leaf_recall": Rec,
            "train_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": leaf_loss, "log": tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, _, _, leaf = batch
        else:
            points, leaf = batch

        pred_leaf_features = self(points)

        if "loss_fn" not in self.hparams or self.hparams["loss_fn"] == "v2":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v3":
            criterion_cluster = SpaceSimilarityLossV3(points)
        elif self.hparams["loss_fn"] == "v4":
            criterion_cluster = SpaceSimilarityLossV4(points)
        elif self.hparams["loss_fn"] == "knn_space_mean":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v5":
            criterion_cluster = SpaceSimilarityLossV5(points)

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetricsTraining(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "val_leaf_loss": leaf_loss,
            "val_leaf_accuracy": Acc,
            "val_leaf_precision": Prec,
            "val_leaf_recall": Rec,
            "val_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return tensorboard_logs

    def validation_epoch_end(self, batch):
        if "real_data" in self.hparams:
            self.validation_real_data()

    def validation_real_data(self):
        real_data_path = self.hparams["real_data"]

        instance_model = self.to(self._device)
        # instance_model.DGCNN_feature_space.device_name = device_name

        files = os.listdir(real_data_path)
        accs = []
        precisions = []
        recals = []
        f1s = []
        pred_images = []

        for file in files:
            path = os.path.join(real_data_path, file)
            main_points, instance_labels, semantic_labels = load_real_ply_with_labels(
                path
            )
            points = main_points[semantic_labels == 1]
            instance_labels = instance_labels[semantic_labels == 1]

            points = torch.tensor(points, dtype=torch.float64).to(self._device)
            if (
                "use_normals" in instance_model.hparams
                and instance_model.hparams["use_normals"]
            ):
                pred_instance_features = instance_model(
                    torch.unsqueeze(points, dim=0).to(self._device)
                )
            else:
                pred_instance_features = instance_model(
                    torch.unsqueeze(points[:, :3], dim=0).to(self._device)
                )

            pred_instance_features = (
                pred_instance_features.cpu().detach().numpy().squeeze()
            )
            clustering = DBSCAN(eps=1, min_samples=10).fit(pred_instance_features)
            pred_final_cluster = clustering.labels_

            d_colors = distinct_colors(len(list(set(pred_final_cluster))))
            colors = np.zeros((pred_final_cluster.shape[0], 3))
            for i, l in enumerate(list(set(pred_final_cluster))):
                colors[pred_final_cluster == l, :] = d_colors[i]

            non_focal_points = main_points[semantic_labels == 2]
            ground_points = main_points[semantic_labels == 0]

            non_focal_color = [0, 0, 0.7, 0.3]
            ground_color = [0.3, 0.1, 0, 0.3]

            metric_calculator = LeafMetrics()
            acc, precison, recal, f1 = metric_calculator(
                torch.tensor(pred_final_cluster).unsqueeze(0).unsqueeze(-1),
                torch.tensor(instance_labels).unsqueeze(0).unsqueeze(-1),
            )

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, c=colors)
            ax.scatter(
                non_focal_points[:, 0],
                non_focal_points[:, 1],
                non_focal_points[:, 2],
                s=1,
                color=non_focal_color,
            )
            ax.scatter(
                ground_points[:, 0],
                ground_points[:, 1],
                ground_points[:, 2],
                s=1,
                color=ground_color,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(
                f"acc: {acc*100:.2f} - precision: {precison:.2f} - recall: {recal:.2f} - f1: {f1:.2f}"
            )
            fig.canvas.draw()
            X = (
                torch.tensor(np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3])
                .transpose(0, 2)
                .transpose(1, 2)
            )
            X = torchvision.transforms.functional.resize(X, (1000, 1000))
            plt.close(fig)
            accs.append(acc)
            precisions.append(precison)
            recals.append(recal)
            f1s.append(f1)
            pred_images.append(X)

        accs = torch.tensor(accs)
        precisions = torch.tensor(precisions)
        recals = torch.tensor(recals)
        f1s = torch.tensor(f1s)

        tensorboard_logs = {
            "test_real_acc": torch.mean(accs),
            "test_real_precision": torch.mean(precisions),
            "test_real_recal": torch.mean(recals),
            "test_real_f1": torch.mean(f1s),
        }

        grid = torch.cat(pred_images, 1)
        self.logger.experiment.add_image(
            "pred_real_data", grid, self.trainer.current_epoch
        )

        for key in tensorboard_logs:
            self.logger.experiment.add_scalar(
                key, tensorboard_logs[key], self.trainer.current_epoch
            )
