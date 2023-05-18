#!/usr/bin/env python
import rospy
import os 
import numpy as np

from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import message_filters

import cv2
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime
import json

path = "/home/mias/Datasets/carla_pothole2"
DISTANCE_CAPTURE = 5.0

TAGS = ["rgb_front",  "semantic_segmentation",  "depth"]
# BASELINES = ["03", "06", "09", "12", "15"]
BASELINES = ["05"]

# ROLLS = ["45", "30", "15", "0", "-15", "-30", "-45"]

with open("src/mias_carla_recorder/config/default.json") as f:
    data = json.load(f)
_r = int(data["objects"][-1]['sensors'][2]['spawn_point']['roll'])
roll = str(abs(_r)).zfill(2)
if _r == 0:
    roll = "0"
elif _r < 0:
    roll = '-' + roll

left_sensors  = [tag + "_left" for tag in TAGS]
right_sensors = [tag + "_right" for tag in TAGS]

folder_time = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
# for folder_name in left_sensors + right_sensors + [a + "_" + b for a in ["disparity_left", "disparity_right"] for b in BASELINES]:
#     os.makedirs(os.path.join(path, folder_time, folder_name))

for folder_name in left_sensors + right_sensors + ["disparity_left", "disparity_right"]:
    for base in BASELINES:
        os.makedirs(os.path.join(path, folder_time, "roll"+roll, "base"+base, folder_name))


def get_item_by_id(data, id):
    for d in data:
        if d["id"] == id:
            return d
    return None

def check_assert(baseline):
    with open("src/mias_carla_recorder/config/default.json") as f:
        data = json.load(f)

    sensors = get_item_by_id(data["objects"], "ego_vehicle")["sensors"]

    L_cfg = get_item_by_id(sensors, "rgb_front_left_" + str(baseline) + "_" + str(roll))
    R_cfg = get_item_by_id(sensors, "rgb_front_right_" + str(baseline) + "_" + str(roll))

    for tag in TAGS:
        l = get_item_by_id(sensors, tag + "_left_" + str(baseline) + "_" + str(roll))
        for k in l["spawn_point"].keys():
            assert l["spawn_point"][k] == L_cfg["spawn_point"][k]
        
        r = get_item_by_id(sensors, tag + "_right_" + str(baseline) + "_" + str(roll))
        for k in l["spawn_point"].keys():
            assert r["spawn_point"][k] == R_cfg["spawn_point"][k]

        # for key in ["image_size_x", "image_size_y", "fov"]:
        assert l["image_size_x"] == r["image_size_x"] == 1280
        assert l["image_size_y"] == r["image_size_y"] == 720
        assert l["fov"] == r["fov"] == 90.0
        
    assert L_cfg["spawn_point"]["x"] == R_cfg["spawn_point"]["x"] # and L_cfg["spawn_point"]["z"] == R_cfg["spawn_point"]["z"]
    return L_cfg, R_cfg

for baseline in BASELINES:
    L_cfg, R_cfg = check_assert(baseline)

    dy = float(L_cfg["spawn_point"]["y"] - R_cfg["spawn_point"]["y"])
    dz = float(L_cfg["spawn_point"]["z"] - R_cfg["spawn_point"]["z"])

    # print(abs((int(baseline) / 10) ** 2 - (dy ** 2 + dz ** 2)))
    err = (float(baseline) / 10) ** 2 - (dy ** 2 + dz ** 2) 
    if not abs(err) < 1e-3:
        print(baseline, dy, dz)
    assert abs(err) < 1e-3

    ImageSizeX, ImageSizeY = L_cfg["image_size_x"], L_cfg["image_size_y"]
    CameraFOV = L_cfg["fov"]

    focal_length = ImageSizeX /(2 * np.tan(CameraFOV * np.pi / 360))
    Center_X = ImageSizeX / 2
    Center_Y = ImageSizeY / 2
    print("baseline,", baseline, "focal_length", focal_length)


bridge = CvBridge()
last_position = {base:None for base in BASELINES}
img_count = {base:0 for base in BASELINES}

def multi_callback(sb_rgb_l, sb_rgb_r, sb_seg_l, sb_seg_r, sb_dep_l, sb_dep_r, sb_odom, base, roll):
    global last_position, img_count
    # print(base, roll)

    tmp = sb_odom.pose.pose.position
    cur_position = np.array([tmp.x, tmp.y, tmp.z])

    if last_position[base] is not None and np.sum((cur_position - last_position[base]) ** 2) < DISTANCE_CAPTURE ** 2:
        return
    last_position[base] = cur_position

    rgb_l = bridge.imgmsg_to_cv2(sb_rgb_l, 'bgr8')
    rgb_r = bridge.imgmsg_to_cv2(sb_rgb_r, 'bgr8')
    seg_l = bridge.imgmsg_to_cv2(sb_seg_l, 'bgr8')
    seg_r = bridge.imgmsg_to_cv2(sb_seg_r, 'bgr8')
    
    scales = np.array([65536.0, 256.0, 1.0, 0]) / (256**3 - 1) * 1000
    dep_l = bridge.imgmsg_to_cv2(sb_dep_l, '8UC4')
    dep_r = bridge.imgmsg_to_cv2(sb_dep_r, '8UC4')

    depth_left = np.dot(dep_l, scales).astype(np.float32)
    disparity_left = (float(base) / 10) * focal_length / depth_left

    depth_right = np.dot(dep_r, scales).astype(np.float32)
    disparity_right = (float(base) / 10) * focal_length / depth_right

    print("同步完成！")
    # cv2.imshow(str(base) + "0", rgb_l)
    # cv2.imshow(str(base) + "1", rgb_r)
    # cv2.imshow(str(base) + "2", seg_l)
    # cv2.imshow(str(base) + "3", seg_r)
    # cv2.imshow(str(base) + "4", depth_left.astype(np.uint8))
    # cv2.imshow(str(base) + "5", disparity_right.astype(np.uint8))
    # cv2.waitKey(1)
    
    name = str(sb_rgb_l.header.stamp.secs) + "_" + str(sb_rgb_l.header.stamp.nsecs)
    # name = str(img_count[base])
    
    print(os.path.join(path, folder_time, "roll"+roll, "base"+base, "rgb_front_left",                name + '.png'))
    cv2.imwrite(os.path.join(path, folder_time, "roll"+roll, "base"+base, "rgb_front_left",                name + '.png'), rgb_l)
    cv2.imwrite(os.path.join(path, folder_time, "roll"+roll, "base"+base, "rgb_front_right",               name + '.png'), rgb_r)
    cv2.imwrite(os.path.join(path, folder_time, "roll"+roll, "base"+base, "semantic_segmentation_left",    name + '.png'), seg_l)
    cv2.imwrite(os.path.join(path, folder_time, "roll"+roll, "base"+base, "semantic_segmentation_right",   name + '.png'), seg_r)
    cv2.imwrite(os.path.join(path, folder_time, "roll"+roll, "base"+base, "depth_left",                    name + '.png'), dep_l)
    cv2.imwrite(os.path.join(path, folder_time, "roll"+roll, "base"+base, "depth_right",                   name + '.png'), dep_r)
    np.save(os.path.join(path, folder_time, "roll"+roll, "base"+base, "disparity_left",  name + '.npy'), disparity_left)
    np.save(os.path.join(path, folder_time, "roll"+roll, "base"+base, "disparity_right", name + '.npy'), disparity_right)
    
    img_count[base] += 1



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    sb_odom = message_filters.Subscriber("/carla/ego_vehicle/odometry", Odometry)

    sb_rgb_l, sb_rgb_r, sb_seg_l, sb_seg_r, sb_dep_l, sb_dep_r = [], [], [], [], [], []
    for b in BASELINES:
        sb_rgb_l.append(message_filters.Subscriber("/carla/ego_vehicle/rgb_front_left_" + b + "_"+ roll + "/image", Image))
        sb_rgb_r.append(message_filters.Subscriber("/carla/ego_vehicle/rgb_front_right_" + b + "_"+ roll + "/image", Image))

        sb_seg_l.append(message_filters.Subscriber("/carla/ego_vehicle/semantic_segmentation_left_" + b + "_"+ roll + "/image", Image))
        sb_seg_r.append(message_filters.Subscriber("/carla/ego_vehicle/semantic_segmentation_right_" + b + "_"+ roll + "/image", Image))

        sb_dep_l.append(message_filters.Subscriber("/carla/ego_vehicle/depth_left_" + b + "_"+ roll + "/image", Image))
        sb_dep_r.append(message_filters.Subscriber("/carla/ego_vehicle/depth_right_" + b + "_"+ roll + "/image", Image))

        sync = message_filters.TimeSynchronizer([sb_rgb_l[-1], sb_rgb_r[-1], sb_seg_l[-1], sb_seg_r[-1], sb_dep_l[-1], sb_dep_r[-1], sb_odom], 10)#同步时间戳，具体参数含义需要查看官方文档。
        sync.registerCallback(multi_callback, (b), (roll))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()