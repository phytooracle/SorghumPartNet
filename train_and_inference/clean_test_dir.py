import os

base = '/home/u9/bhuppenthal/xdisk/vrbio/test/segmentation_pointclouds'
ids = os.listdir(base)
for id in ids:
    files = os.listdir(os.path.join(base, id))
    for file in files:
        if file == 'combined_multiway_registered.ply':
            continue
        os.remove(os.path.join(base, id, file))