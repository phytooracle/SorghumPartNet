import os
import numpy as np
import torch
import open3d as o3d
import torch.nn.functional as F
import sys
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import time
import traceback
import logging

sys.path.append("..")
# from models.nn_models import *
from models.reduced import *
from data.utils import create_ply_pcd_from_points_with_labels
from data.load_raw_data import (
    load_ply_file_points
)
from scipy.spatial.distance import cdist
from scipy import stats
import argparse

device = None
device_name = None
args = None

logging.basicConfig(
    stream=sys.stderr,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Sorghum 3D part segmentation prediction script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '-m',
        '--model',
        help='Path to the folder containing the models (semantic and instance).',
        metavar='model',
        required=False,
        type=str,
        default='/home/u9/bhuppenthal/models/PlantSegNet'
    )

    parser.add_argument(
        "-s",
        "--semantic_version",
        help="The version of the semantic model. If not determined, the latest version would be picked.",
        metavar="semantic_version",
        required=False,
        type=int,
        default=-1,
    )

    parser.add_argument(
        "-i",
        "--instance_version",
        help="The version of the instance model. If not determined, the latest version would be picked.",
        metavar="instance_version",
        required=False,
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "-f",
        "--full_size",
        help="Whether to run the prediction on the full size point cloud.",
        metavar="full_size",
        required=False,
        type=bool,
        default=True,
    )

    parser.add_argument(
        '-n',
        '--num_points',
        help='Number of points to save for the full point cloud.',
        metavar='num_points',
        type=int,
        default=50000
    )

    parser.add_argument(
        '-S',
        '--save_all',
        help="Whether to save the downsampled point clouds.",
        metavar='save_all',
        type=bool,
        default=False
    )

    parser.add_argument(
        "-d",
        "--dist",
        help="Distance threshold below which points are considered to be in the same cluster.",
        metavar="dist",
        required=False,
        type=float,
        default=5,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to the data folder.",
        metavar="path",
        required=False,
        type=str,
        default='/home/u9/bhuppenthal/xdisk/vrbio/test/segmentation_pointclouds'
    )

    parser.add_argument(
        "-t",
        "--type",
        help="Point cloud type, real or synthetic. 0 means synthetic and 1 means real.",
        metavar="type",
        required=False,
        type=int,
        default=0,
    )

    parser.add_argument(
        "-c",
        "--cluster_method",
        help="The clustering method to be used.",
        metavar="method",
        required=False,
        type=str,
        default="dbscan",
    )

    parser.add_argument(
        '-T',
        '--time',
        help='Time PlantSegNet per plant.',
        metavar='time',
        type=bool,
        default=False
    )

    return parser.parse_args()


def predict_downsampled(points, semantic_model, instance_model, method, dist=5):
    if (
        "use_normals" in semantic_model.hparams
        and semantic_model.hparams["use_normals"]
    ):
        pred_semantic_label = semantic_model(torch.unsqueeze(points, dim=0).to(device))
    else:
        pred_semantic_label = semantic_model(torch.unsqueeze(points[:,:3], dim=0).to(device))

    pred_semantic_label = F.softmax(pred_semantic_label, dim=1)
    pred_semantic_label = pred_semantic_label[0].cpu().detach().numpy().T
    pred_semantic_label_labels = np.argmax(pred_semantic_label, 1)

    instance_points = points[pred_semantic_label_labels == 1,:]

    if (
        "use_normals" in instance_model.hparams
        and instance_model.hparams["use_normals"]
    ):
        pred_instance_features = instance_model(torch.unsqueeze(instance_points, dim=0).to(device))
    else:
        pred_instance_features = instance_model(torch.unsqueeze(instance_points[:,:3], dim=0).to(device))

    if method == "dbscan":
        pred_instance_features = pred_instance_features.cpu().detach().numpy().squeeze()
        clustering = DBSCAN(eps=1, min_samples=10).fit(pred_instance_features)
        pred_final_cluster = clustering.labels_
    else:
        distance_pred = torch.cdist(pred_instance_features, pred_instance_features)
        distance_pred = distance_pred.cpu().detach().numpy().T
        distance_pred = np.squeeze(distance_pred)

        distance_pred = 1 * (distance_pred < dist)

        pred_final_cluster = np.zeros((distance_pred.shape[0])).astype("uint16")
        next_label = 1

        for i in range(distance_pred.shape[0]):
            if pred_final_cluster[i] == 0:
                pred_final_cluster[i] = next_label
                next_label += 1

            ind = np.where(distance_pred[i] == 1)

            for j in ind[0]:
                pred_final_cluster[j] = pred_final_cluster[i]

    # print("Number of predicted leaf instances: ", len(list(set(pred_final_cluster))))

    points = points.cpu().detach().numpy()
    instance_points = instance_points.cpu().detach().numpy()

    ply_semantic = create_ply_pcd_from_points_with_labels(
        points[:, :3], pred_semantic_label_labels, is_semantic=True
    )

    ply_instance = create_ply_pcd_from_points_with_labels(
        instance_points[:, :3], pred_final_cluster
    )

    return ply_semantic, ply_instance, pred_semantic_label_labels, pred_final_cluster


def pred_full_size(
    full_points, downsampled_points, downsampled_semantic, downsampled_instance, k=10
):

    distances = cdist(full_points, downsampled_points)
    full_ind, down_ind = np.where(distances == 0)

    semantic_full = np.ones((full_points.shape[0])).astype("uint8") * -1
    semantic_full[full_ind] = downsampled_semantic[down_ind]

    for i in range(full_points.shape[0]):
        sorted_distances = np.argsort(distances[i])
        if semantic_full[i] == -1:
            mode = stats.mode(downsampled_semantic[sorted_distances[:k]])[0]
            semantic_full[i] = mode

    semantic_ply = create_ply_pcd_from_points_with_labels(
        full_points, semantic_full, is_semantic=True
    )

    downsampled_focal_points = downsampled_points[downsampled_semantic == 1]
    focal_points = full_points[semantic_full == 1]

    distances = cdist(focal_points, downsampled_focal_points)
    full_ind, down_ind = np.where(distances == 0)

    instance_full = np.ones((focal_points.shape[0])).astype("uint8") * -1
    instance_full[full_ind] = downsampled_instance[down_ind]

    for i in range(focal_points.shape[0]):
        sorted_distances = np.argsort(distances[i])
        if instance_full[i] == -1:
            mode = stats.mode(downsampled_instance[sorted_distances[:k]])[0]
            instance_full[i] = mode

    instance_ply = create_ply_pcd_from_points_with_labels(focal_points, instance_full)

    return semantic_ply, instance_ply


def save_predicted(ply_pcd, path):
    o3d.io.write_point_cloud(path, ply_pcd)


def load_model_chkpoint(model, path):

    model = eval(model).load_from_checkpoint(path)
    model = model.to(device)
    # print(model.hparams)
    # print(model.state_dict()['scale'])
    # print(model.state_dict()['threshold'])
    model.eval()
    return model


def load_model(model_name, version):
    if version == -1:
        versions = os.listdir(
            os.path.join(f'{args.model}', model_name)
        )
        version = sorted(versions)[-1].split("_")[-1]

    path_all_checkpoints = os.path.join(f'{args.model}', model_name, f'lightning_logs/version_{version}/checkpoints')
    path = sorted(os.listdir(path_all_checkpoints))[-1]
    print(f"{model_name} using version ", version, " ", path)

    model = load_model_chkpoint(model_name, os.path.join(path_all_checkpoints, path))
    return model


def main_ply():

    semantic_model = load_model("SorghumPartNetSemantic", args.semantic_version).double()
    instance_model = load_model("SorghumPartNetInstance", args.instance_version).double()

    folder_names = sorted(os.listdir(args.path))
    times = []

    for folder in tqdm(folder_names, file=sys.stdout):

        if args.time:
            time_per_pointcloud = time.time()

        base_path = os.path.join(args.path, folder)
        path_to_pcd = os.path.join(base_path, 'combined_multiway_registered.ply')

        # check if already processed
        if len(os.listdir(base_path)) > 1:
            continue

        # print(f':: Opening {path_to_pcd}')
        try:
            points_full, points, normals = load_ply_file_points(path_to_pcd, n_points=8000, full_points=8000)
        except Exception as error:
            tb = traceback.format_exc()
            logging.error('Exception occurred:\n%s', tb)
            continue

        points = torch.tensor(points, dtype=torch.float64)
        normals = torch.tensor(normals, dtype=torch.float64)
        points_full = torch.tensor(points_full, dtype=torch.float64)
        points = torch.cat((points, normals), -1)
        # print(f":: Point cloud {path_to_pcd} with {points_full.shape[0]} points loaded!")

        try:
            # print(':: Running predict_downsampled...')
            (
                downsampled_semantic_pcd,
                downsampled_instance_pcd,
                downsampled_semantics,
                downsampled_instance,
            ) = predict_downsampled(points, semantic_model, instance_model, args.cluster_method)

            # print(':: Running pred_full_size...')
            semantic_pcd, instance_pcd = pred_full_size(
                points_full, points[:, :3], downsampled_semantics, downsampled_instance
            )
        except Exception as error:
            tb = traceback.format_exc()
            logging.error('Exception occurred:\n%s', tb)
            continue

        # print(':: Saving the point clouds...')
        if args.save_all:
            save_predicted(
                downsampled_semantic_pcd,
                os.path.join(base_path, 'semantic.ply')
            )
            save_predicted(
                downsampled_instance_pcd,
                os.path.join(base_path, 'instance.ply')
            )
        save_predicted(
            semantic_pcd,
            os.path.join(base_path, 'semantic_full.ply')
        )
        save_predicted(
            instance_pcd,
            os.path.join(base_path, 'instance_full.ply')
        )

        if args.time:
            time_per_pointcloud = time.time() - time_per_pointcloud
            times.append(time)
    
    if args.time:
        time_per_pointcloud = np.asarray(time_per_pointcloud)
        mean = np.mean(time_per_pointcloud)
        stdev = np.std(time_per_pointcloud)
        print(f'mean: {mean}, std: {stdev}')

        num_in_july = len(os.listdir(os.path.expanduser('~/xdisk/vrbio/2020-07-30/segmentation_pointclouds')))
        num_in_aug = len(os.listdir(os.path.expanduser('~/xdisk/vrbio/2020-08-03/segmentation_pointclouds')))
        print(f'estimated time for july: {num_in_july * mean}')
        print(f'estimated time for aug: {num_in_aug * mean}')


def main():
    global args
    args = get_args()

    global device_name
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    global device
    device = torch.device(device_name)

    print(f'Running on {device_name}')

    main_ply()


if __name__ == "__main__":
    main()
