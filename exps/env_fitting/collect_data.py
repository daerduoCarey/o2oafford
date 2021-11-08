"""
    Init an env and collect data
"""

import os
import sys
import shutil
import random
import numpy as np
from PIL import Image
import cv2
import json
from argparse import ArgumentParser
from pyquaternion import Quaternion

from sapien.core import Pose
from env import Env, ContactError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from robots.panda_robot import Robot
from camera import Camera
from utils import get_global_position_from_camera, save_h5, get_random_number, printout, get_random_z_up_rot_quat

from geometry_utils import export_pts, sample_points

parser = ArgumentParser()
parser.add_argument('shape_id', type=str)
parser.add_argument('category', type=str)
parser.add_argument('--out_dir', type=str, default='./results/env_fitting')
parser.add_argument('--epoch_id', type=int, default=0, help='epoch id')
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--data_split', type=str, default='test_cat')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()

out_dir = os.path.join(args.out_dir, '%s_%s_%d_%d' % (args.shape_id, args.category, args.epoch_id, args.trial_id))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict()

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

#  load cat-freq
cat2freq = dict()
with open('../stats/all_cats_cnt_freq.txt', 'r') as fin:
    for l in fin.readlines():
        cat, _, freq = l.rstrip().split()
        cat2freq[cat] = int(freq)

# load act obj list
cats_train_test_split = 'train' if 'train_cat' in args.data_split else 'test'
with open(os.path.join(BASE_DIR, 'stats', 'act_cats-%s.txt' % cats_train_test_split), 'r') as fin:
    act_cats = [l.rstrip() for l in fin.readlines()]
act_shapes = []
for act_cat in act_cats:
    with open('../stats/%s-%s.txt' % (act_cat, args.data_split), 'r') as fin:
        for l in fin.readlines():
            act_shape = l.rstrip()
            act_shapes += [act_shape] * cat2freq[act_cat]

# setup env
env = Env(flog=flog, show_gui=(not args.no_gui))

# setup camera
cam = Camera(env, random_front_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# load shape
object_material = env.get_material(4, 4, 0.01)
env.set_default_material(object_material)
# table
scene_object_joint_angles = env.load_scene_object(args.shape_id)
# get z_max as the upper limit for placement
vs, _ = env.get_global_mesh(env.scene_object)
z_max = float(np.max(vs[:, 2])) - 0.05
out_info['scene_object'] = args.shape_id
out_info['scene_object_z_max'] = z_max
out_info['scene_object_joint_angles'] = scene_object_joint_angles

env.drive_scene_object_qpos_to_starting_limits(scene_object_joint_angles)

# simulate some steps for the objects to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    scene_object_qpos = env.get_scene_object_qpos()
    dist = np.linalg.norm(scene_object_qpos - scene_object_joint_angles)
    if dist < 1e-3:
        still_timesteps += 1
    else:
        still_timesteps = 0
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Scene Object cannot remain at the starting articulated part poses!\n')
    flog.close()
    env.close()
    exit(1)


### check the object parts can be closed without any object 
# Sometimes, the object parts cannot be closed due to the inter-part collision by itself
# We need to remove these bad simulation trials
scene_object_qpos_lower_limits = env.drive_scene_object_qpos_to_lower_limits()

still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    scene_object_qpos = env.get_scene_object_qpos()
    dist = np.max(np.abs(scene_object_qpos - scene_object_qpos_lower_limits))
    if dist < 1e-3:
        still_timesteps += 1
    else:
        still_timesteps = 0
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Scene Object cannot be fully closed from the starting articulated part poses!\n')
    flog.close()
    env.close()
    exit(1)

# re-open the starting parts
env.drive_scene_object_qpos_to_starting_limits(scene_object_joint_angles)

# simulate some steps for the objects to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    scene_object_qpos = env.get_scene_object_qpos()
    dist = np.linalg.norm(scene_object_qpos - scene_object_joint_angles)
    if dist < 1e-3:
        still_timesteps += 1
    else:
        still_timesteps = 0
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Scene Object cannot remain at the starting articulated part poses!\n')
    flog.close()
    env.close()
    exit(1)


### use the GT vision
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
#export_pts(os.path.join(out_dir, 'scene_cambase.pts'), cam_XYZA_pts @ cam.get_metadata()['cam2cambase'][:3, :3].T)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
save_h5(os.path.join(out_dir, 'cam_XYZA.h5'), \
        [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'), \
         (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'), \
         (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'), \
        ])

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

all_position_cam = cam_XYZA[:, :, :3].reshape(-1, 3)
all_1s = np.ones((all_position_cam.shape[0], 1), dtype=np.float32)
all_position_cam_xyz1 = np.concatenate([all_position_cam, all_1s], axis=1)
all_position_world_xyz1 = all_position_cam_xyz1 @ cam.get_metadata()['mat44'].T
all_position_world_xyz1 = all_position_world_xyz1[:, 2].reshape(rgb.shape[0], rgb.shape[1])
all_position_world_valid = all_position_world_xyz1 < z_max

# randomly pick one from the pixeles with possible normal direction (almost along with z-axis)
# convert to world space
gt_nor = gt_nor[:, :, :3]
gt_nor = (gt_nor.reshape(-1, 3) @ cam.get_metadata()['mat44'][:3, :3].T).reshape(rgb.shape[0], rgb.shape[1], 3)
gt_nor_valid = gt_nor[:, :, 2] > 0.95

final_valid = gt_nor_valid * all_position_world_valid
Image.fromarray(final_valid.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_possible.png'))
Image.fromarray(all_position_world_valid.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_applicable.png'))

# sample one pixel to interact
xs, ys = np.where(final_valid)
if len(xs) == 0:
    printout(flog, 'No Possible Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]
out_info['pixel_locs'] = [int(x), int(y)]
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(out_dir, 'point_to_interact.png'))

# compute the intended dropping pixel link
id2link = dict()
for l in env.scene_object.get_links():
    id2link[l.get_id()] = l
link_mask = cam.get_link_mask()
dropping_pixel_link = id2link[link_mask[x, y]]
#print('dropping_pixel_link: ', dropping_pixel_link)

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
out_info['position_cam'] = position_cam.tolist()
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
out_info['position_world'] = position_world.tolist()

# load acting object
init_x = position_world[0]
init_y = position_world[1]
init_z = position_world[2]
init_scale = get_random_number(0.05, 0.3)
acting_object_shape_id = np.random.choice(act_shapes)
acting_object_setting = {'name': 'object', 'shape_id': acting_object_shape_id, \
        'fix_root_link': False, 'state': 'locked', \
        'init_z': init_z, 'init_x': init_x, 'init_y': init_y, 'init_scale': init_scale, \
        'init_quat': get_random_z_up_rot_quat()}
acting_object_setting = env.load_acting_object(acting_object_setting, is_putting=True)
out_info['acting_object'] = acting_object_setting
env.render()
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_start.png'))

# export final acting object collision mesh (cambase-space, zero-centered)
vs, fs = env.get_global_mesh(env.acting_object)
acting_object_center = env.acting_object.get_root_pose().p
vs[:, 0] -= acting_object_center[0]
vs[:, 1] -= acting_object_center[1]
vs[:, 2] -= acting_object_center[2]
acting_object_pts, _ = sample_points(vs, fs+1, num_points=1000)
acting_object_pts = acting_object_pts @ cam.get_metadata()['base_mat44'][:3, :3]
export_pts(os.path.join(out_dir, 'acting_object_cambase.pts'), acting_object_pts)


### wait to start
if not args.no_gui:
    ### wait to start
    env.wait_to_start()


### main steps
success = True

# check: have no collision with the scene objects at beginning; 
acting_object_lids = [l.get_id() for l in env.acting_object.get_links()]
env.step()
for c in env.scene.get_contacts():
    aid1 = c.actor1.get_id()
    aid2 = c.actor2.get_id()
    has_impulse = False
    for p in c.points:
        if abs(p.impulse @ p.impulse) > 1e-4:
            has_impulse = True
            break
    if has_impulse:
        if (aid1 in acting_object_lids) and (aid2 not in acting_object_lids):
            printout(flog, '[FAIL] collision at start!\n')
            success = False
            break
        if (aid2 in acting_object_lids) and (aid1 not in acting_object_lids):
            printout(flog, '[FAIL] collision at start!\n')
            success = False
            break

# drop it off stably
# pose should not change too much, except center
# check staying on table
if success:
    start_center, start_quat = env.get_acting_object_pose()
    start_quat = Quaternion(start_quat)
    for _ in range(2000):
        env.step()
        env.render()
        cur_center, cur_quat = env.get_acting_object_pose()
        cur_quat = Quaternion(cur_quat)
        start_angle_diff = Quaternion.absolute_distance(cur_quat, start_quat)/np.pi*180
        if start_angle_diff > 5 or \
                np.max(np.abs(cur_center[:2] - start_center[:2])) > 5e-3 or \
                cur_center[2] < env.delete_thres:
            printout(flog, '[FAIL] pose change or fall off table during falloff: first 2000 steps!\n')
            success = False
            break

# check staying stably
# check staying on table
if success:
    start_center, start_quat = env.get_acting_object_pose()
    start_quat = Quaternion(start_quat)
    for _ in range(10000):
        env.step()
        env.render()
        cur_center, cur_quat = env.get_acting_object_pose()
        cur_quat = Quaternion(cur_quat)
        start_angle_diff = Quaternion.absolute_distance(cur_quat, start_quat)/np.pi*180
        if start_angle_diff > 2 or \
                np.max(np.abs(cur_center - start_center)) > 1e-4 or \
                cur_center[2] < env.delete_thres:
            printout(flog, '[FAIL] pose change or fall off table during falloff: final 10000 steps!\n')
            success = False
            break

# stitch the acting object to the dropping_link
# check: we can successfully close all articulated parts to the lowest limits
if success:
    env.stitch_acting_object(dropping_pixel_link, position_world)

    scene_object_qpos_lower_limits = env.drive_scene_object_qpos_to_lower_limits()
    
    still_timesteps = 0
    wait_timesteps = 0
    while still_timesteps < 5000 and wait_timesteps < 20000:
        env.step()
        env.render()
        scene_object_qpos = env.get_scene_object_qpos()
        dist = np.max(np.abs(scene_object_qpos - scene_object_qpos_lower_limits))
        if dist < 1e-4:
            still_timesteps += 1
        else:
            still_timesteps = 0
        wait_timesteps += 1

    if still_timesteps < 5000:
        printout(flog, '[FAIL] cannot close the articulated parts!\n')
        success = False

rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_final.png'))

out_info['result'] = success

# save results
with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
    json.dump(out_info, fout)

#close the file
flog.close()

if args.no_gui:
    # close env
    env.close()
else:
    if success:
        print('[Successful Interaction] Done. Ctrl-C to quit.')
        while True:
            env.step()
            env.render()
    else:
        print('[Unsuccessful Interaction] Failed. Ctrl-C to quit.')
        # close env
        env.close()

