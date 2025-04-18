import torch
import numpy as np
import random
import colorsys
import open3d as o3d
from scipy.spatial.distance import cdist


def distinct_colors(n=200):

    colors = []
    for i in np.arange(0, 1, 1.0 / n):
        hue = i
        saturation = 1
        lightness = 0.5

        c = colorsys.hsv_to_rgb(hue, saturation, lightness)

        colors.append(c)

    return colors


def rgb_dec_to_hex(tup):
    """
    Given input in [0,1] x 3 indicating RGB color, returns integer hex representation. For use with
    k3d visualization which requires this particular color format.

    ex, (1, 1, 1) -> 255255255
    """
    a = np.array(tup)
    a = (a * 256).astype("uint8")
    return a[0] * 255**2 + a[1] * 255 + a[2]


def get_initial_cluster_assignment(points, samples, labels):
    sample_coords = points[samples]
    sample_labels = labels[samples]
    dist = cdist(points, sample_coords)

    labels = np.expand_dims(labels, -1)
    sample_labels = np.expand_dims(sample_labels, -1)

    dist_labels = cdist(labels, sample_labels)
    dist_labels[dist_labels > 0] = sys.maxsize
    dist += dist_labels
    initial_cluster = np.argmin(dist, 1)
    return initial_cluster


def get_affinity_matrix(labels, samples):
    labels_samples = labels[samples]
    n = samples.shape[0]
    affinity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels_samples[i] == labels_samples[j]:
                affinity_matrix[i, j] = 1
    return affinity_matrix


def get_is_focal_plant(index):
    is_focal = np.zeros(index.shape)
    is_focal[index > 0] = 1
    return is_focal.squeeze()


def create_ply_pcd_from_points_with_labels(points, labels, is_semantic=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 0])
    colors = np.array(pcd.colors)

    if is_semantic:
        d_colors = [[0.3, 0.1, 0], [0, 0.7, 0], [0, 0, 0.7]]
    else:
        d_colors = distinct_colors(len(list(set(labels))))
        random.shuffle(d_colors)

    for i, l in enumerate(list(set(labels))):
        colors[labels == l, :] = d_colors[i]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
