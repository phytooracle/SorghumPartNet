import argparse
import logging
import os
import sys
import traceback

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from data.load_raw_data import load_ply_file_points
from data.utils import create_ply_pcd_from_points_with_labels
from models.reduced import SorghumPartNetInstance, SorghumPartNetSemantic


def get_args():
    """Get the arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Sorghum 3D part segmentation prediction script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Path to the folder containing the models (semantic and instance).",
        metavar="model",
        required=True,
        type=str,
        default="/groups/dukepauli/shared/models/PlantSegNet/",
    )

    parser.add_argument(
        "-s",
        "--semantic_version",
        help="The version of the semantic model. "
        "If not determined, the latest version would be picked.",
        metavar="semantic_version",
        required=False,
        type=int,
        default=-1,
    )

    parser.add_argument(
        "-i",
        "--instance_version",
        help="The version of the instance model. "
        "If not determined, the latest version would be picked.",
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
        "-n",
        "--num_points",
        help="Number of points to save for the full point cloud.",
        metavar="num_points",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "-S",
        "--save_all",
        help="Whether to save the downsampled point clouds.",
        metavar="save_all",
        type=bool,
        default=False,
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
        required=True,
        type=str,
    )  ### Point to the segmentation_pointclouds directory

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
        choices=["dbscan", "distance"],
        type=str,
        default="dbscan",
    )  # "distance" label is a placeholder for the distance-based clustering method

    return parser.parse_args()


def predict_downsampled(
    points, semantic_model, instance_model, method, dist=5
):
    """Predict the downsampled point cloud."""
    if (
        "use_normals" in semantic_model.hparams
        and semantic_model.hparams["use_normals"]
    ):
        pred_semantic_label = semantic_model(
            torch.unsqueeze(points, dim=0).to(semantic_model.device)
        )
    else:
        pred_semantic_label = semantic_model(
            torch.unsqueeze(points[:, :3], dim=0).to(semantic_model.device)
        )

    pred_semantic_label = F.softmax(pred_semantic_label, dim=1)
    pred_semantic_label = pred_semantic_label[0].cpu().detach().numpy().T
    pred_semantic_label_labels = np.argmax(pred_semantic_label, 1)

    instance_points = points[pred_semantic_label_labels == 1, :]

    if (
        "use_normals" in instance_model.hparams
        and instance_model.hparams["use_normals"]
    ):
        pred_instance_features = instance_model(
            torch.unsqueeze(instance_points, dim=0).to(instance_model.device)
        )
    else:
        pred_instance_features = instance_model(
            torch.unsqueeze(instance_points[:, :3], dim=0).to(
                instance_model.device
            )
        )

    if method == "dbscan":
        pred_instance_features = (
            pred_instance_features.cpu().detach().numpy().squeeze()
        )
        clustering = DBSCAN(eps=1, min_samples=10).fit(pred_instance_features)
        pred_final_cluster = clustering.labels_
    else:
        distance_pred = torch.cdist(
            pred_instance_features, pred_instance_features
        )
        distance_pred = distance_pred.cpu().detach().numpy().T
        distance_pred = np.squeeze(distance_pred)

        distance_pred = 1 * (distance_pred < dist)

        pred_final_cluster = np.zeros((distance_pred.shape[0])).astype(
            "uint16"
        )
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

    return (
        ply_semantic,
        ply_instance,
        pred_semantic_label_labels,
        pred_final_cluster,
    )


def pred_full_size(
    full_points,
    downsampled_points,
    downsampled_semantic,
    downsampled_instance,
    device,
    k=10,
):
    """Predict the full size point cloud."""
    # full_points = torch.tensor(full_points, device=device)
    # downsampled_points = torch.tensor(downsampled_points, device=device)
    downsampled_semantic = torch.tensor(downsampled_semantic, device=device)
    downsampled_instance = torch.tensor(downsampled_instance, device=device)

    distances = torch.cdist(full_points, downsampled_points)
    full_ind, down_ind = torch.where(distances == 0)

    semantic_full = torch.full(
        (full_points.shape[0],),
        -1,
        dtype=downsampled_semantic.dtype,
        device=device,
    )
    semantic_full[full_ind] = downsampled_semantic[down_ind]

    for i in range(full_points.shape[0]):
        _, sorted_indices = torch.sort(distances[i])
        if semantic_full[i] == -1:
            nearest_labels = downsampled_semantic[sorted_indices[:k]].view(-1)
            labels, counts = torch.unique(nearest_labels, return_counts=True)
            mode = labels[torch.argmax(counts)]
            semantic_full[i] = mode

    semantic_ply = create_ply_pcd_from_points_with_labels(
        full_points.cpu().numpy(),
        semantic_full.cpu().numpy(),
        is_semantic=True,
    )

    downsampled_focal_points = downsampled_points[downsampled_semantic == 1]
    focal_points = full_points[semantic_full == 1]

    distances = torch.cdist(focal_points, downsampled_focal_points)
    full_ind, down_ind = torch.where(distances == 0)

    instance_full = torch.full(
        (focal_points.shape[0],),
        -1,
        dtype=downsampled_instance.dtype,
        device=device,
    )
    instance_full[full_ind] = downsampled_instance[down_ind]

    for i in range(focal_points.shape[0]):
        _, sorted_indices = torch.sort(distances[i])
        if instance_full[i] == -1:
            nearest_labels = downsampled_instance[sorted_indices[:k]]
            labels, counts = torch.unique(nearest_labels, return_counts=True)
            mode = labels[torch.argmax(counts)]
            instance_full[i] = mode

    instance_ply = create_ply_pcd_from_points_with_labels(
        focal_points.cpu().numpy(), instance_full.cpu().numpy()
    )

    return semantic_ply, instance_ply


def save_predicted(ply_pcd, path):
    """Save the predicted point cloud to a file."""
    o3d.io.write_point_cloud(path, ply_pcd)


def load_model_chkpoint(model, path, device):
    """Load the model from a checkpoint."""
    model = eval(model).load_from_checkpoint(path, device=device)
    model = model.to(device)
    # print(model.hparams)
    # print(model.state_dict()['scale'])
    # print(model.state_dict()['threshold'])
    model.eval()
    return model


def load_model(args, model_name, version, device):
    """Load the model from a checkpoint."""
    if version == -1:
        versions = os.listdir(os.path.join(f"{args.model}", model_name))
        version = sorted(versions)[-1].split("_")[-1]

    path_all_checkpoints = os.path.join(
        f"{args.model}",
        model_name,
        f"lightning_logs/version_{version}/checkpoints",
    )
    path = sorted(os.listdir(path_all_checkpoints))[-1]
    print(f"{model_name} using version ", version, " ", path)

    model = load_model_chkpoint(
        model_name, os.path.join(path_all_checkpoints, path), device
    )
    return model


def worker(args, cpu_id, device, ids):
    """This is where the cpu worker starts."""
    logging.basicConfig(
        # stream=sys.stderr,
        filename=f"output-{cpu_id}.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print(f"hello from worker {cpu_id} using {device}!")

    semantic_model = load_model(
        args, "SorghumPartNetSemantic", args.semantic_version, device
    ).double()
    instance_model = load_model(
        args, "SorghumPartNetInstance", args.instance_version, device
    ).double()

    for id in ids:

        base_path = os.path.join(args.path, id)
        path_to_pcd = os.path.join(
            base_path, "combined_multiway_registered.ply"
        )

        # check if already processed
        if len(os.listdir(base_path)) > 1:
            continue

        # print(f':: Opening {path_to_pcd}')
        try:
            points_full, points, normals = load_ply_file_points(
                path_to_pcd, n_points=8000, full_points=8000
            )
        except Exception:
            tb = traceback.format_exc()
            logging.error("Exception occurred:\n%s", tb)
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
            ) = predict_downsampled(
                points, semantic_model, instance_model, args.cluster_method
            )

            # print(':: Running pred_full_size...')
            semantic_pcd, instance_pcd = pred_full_size(
                points_full,
                points[:, :3],
                downsampled_semantics,
                downsampled_instance,
                device,
            )
        except Exception:
            tb = traceback.format_exc()
            logging.error("Exception occurred:\n%s", tb)
            continue

        # print(':: Saving the point clouds...')
        if args.save_all:
            save_predicted(
                downsampled_semantic_pcd,
                os.path.join(base_path, "semantic.ply"),
            )
            save_predicted(
                downsampled_instance_pcd,
                os.path.join(base_path, "instance.ply"),
            )
        save_predicted(
            semantic_pcd, os.path.join(base_path, "semantic_full.ply")
        )
        save_predicted(
            instance_pcd, os.path.join(base_path, "instance_full.ply")
        )


if __name__ == "__main__":
    args = get_args()

    if not torch.cuda.is_available():
        print(":: no gpu!")
        sys.exit()

    num_gpus = torch.cuda.device_count()

    ids = np.asarray(os.listdir(args.path))
    split_ids = np.split(ids, num_gpus)

    assignments = [
        (i, torch.device(f"cuda:{i}"), split_ids[i]) for i in range(num_gpus)
    ]
    procs = [
        mp.Process(target=worker, args=(args, cpu_id, device, ids))
        for cpu_id, device, ids in assignments
    ]
    mp.set_start_method("spawn")

    for proc in procs:
        proc.start()

    try:
        for proc in procs:
            proc.join()  # Wait for all processes to finish
    except KeyboardInterrupt:
        print("Main program received interrupt. Terminating workers...")
        for proc in procs:
            proc.terminate()  # Forcefully terminate processes
        for proc in procs:
            proc.join()  # Ensure processes are cleaned up

    print("multiprocessing finished")
