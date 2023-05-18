import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import open3d as o3d
import numpy as np
import time

path = "/home/mias/Datasets/carla_pothole"

p_rgb = "../sample_data/rgb.png"
p_dpt = '../sample_data/depth.png'

if not os.path.exists(p_rgb) or not os.path.exists(p_dpt):
    print("image not exist")

img = cv2.imread(p_rgb)
arr = cv2.imread(p_dpt)

scales = np.array([65536.0, 256, 1]) / (256**3 - 1) * 1000
depth_image = np.dot(arr, scales).astype(np.float32)

# vis = o3d.visualization.Visualizer()

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()

######## 未加几何 #########
# points = [[j, i, depth_image[i][j]] for j in range(depth_image.shape[1]) for i in range(depth_image.shape[0])]
points = []
colors = []
for i in range(depth_image.shape[0]):
    for j in range(depth_image.shape[1]):
        if depth_image[i][j] > 999:
            continue
        points.append([(i-depth_image.shape[0]/2)*depth_image[i][j]/1000, 
                       (j-depth_image.shape[1]/2)*depth_image[i][j]/1000, 
                       depth_image[i][j]])
        colors.append(img[i][j])
        
#################

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)


vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
o3d.visualization.ViewControl.set_front(vis.get_view_control(), [0,0,-1])
o3d.visualization.ViewControl.set_up(vis.get_view_control(), [-1,0,0])

vis.run()
# o3d.visualization.draw_geometries([pcd], window_name="depth")

