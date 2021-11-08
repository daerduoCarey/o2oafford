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
parser.add_argument('--out_dir', type=str, default='./results/env_pushing')
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
cam = Camera(env, random_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# material
object_material = env.get_material(4, 4, 0.01)
env.set_default_material(object_material)
# scene object
env.load_scene_object(args.shape_id)
out_info['scene_object'] = args.shape_id
env.render()

# simulate some steps for the scene object to fall down to the ground and stay rest
still_timesteps = 0
wait_timesteps = 0
cur_center, cur_quat = env.get_scene_object_pose()
cur_quat = Quaternion(cur_quat)
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    new_center, new_quat = env.get_scene_object_pose()
    new_quat = Quaternion(new_quat)
    angle_diff = Quaternion.absolute_distance(cur_quat, new_quat)/np.pi*180
    if angle_diff < 1e-2 and np.max(np.abs(cur_center[:2] - new_center[:2])) < 1e-5:
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_center = new_center
    cur_quat = new_quat
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Scene Object Not Still!\n')
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

# output applicable map
gt_applicable = np.zeros((cam.image_size, cam.image_size), dtype=np.bool)
gt_applicable[cam_XYZA_id1, cam_XYZA_id2] = True
Image.fromarray(gt_applicable.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_applicable.png'))
Image.fromarray(gt_applicable.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_possible.png'))

# sample one pixel to interact
xs, ys = np.where(gt_applicable)
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

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
out_info['position_cam'] = position_cam.tolist()
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
out_info['position_world'] = position_world.tolist()

# compute cambase front/x-direction
cambase_xdir_world = cam.get_metadata()['base_mat44'][:3, 0]

# load acting object
init_x = position_world[0]
init_y = position_world[1]
init_z = position_world[2]
init_scale = get_random_number(0.3, 1.5)
acting_object_shape_id = np.random.choice(act_shapes)
acting_object_setting = {'name': 'object', 'shape_id': acting_object_shape_id, \
        'fix_root_link': False, 'state': 'locked', \
        'init_z': init_z, 'init_x': init_x, 'init_y': init_y, 'init_scale': init_scale, \
        'init_quat': get_random_z_up_rot_quat()}
acting_object_setting = env.load_acting_object(acting_object_setting, \
        cambase_xdir_world=cambase_xdir_world, is_putting=True)
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
if success:
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

# drive acting object to move short-term along cambase_xdir
if success:
    env.drive_acting_object(cambase_xdir_world)
    start_center, start_quat = env.get_scene_object_pose()
    start_quat = Quaternion(start_quat)

    # simulate some steps for the scene object to stay rest
    still_timesteps = 0
    wait_timesteps = 0
    cur_center, cur_quat = env.get_scene_object_pose()
    cur_quat = Quaternion(cur_quat)
    while still_timesteps < 5000 and wait_timesteps < 20000:
        env.step()
        env.render()
        new_center, new_quat = env.get_scene_object_pose()
        new_quat = Quaternion(new_quat)
        angle_diff = Quaternion.absolute_distance(cur_quat, new_quat)/np.pi*180
        if angle_diff < 1e-2 and np.max(np.abs(cur_center[:2] - new_center[:2])) < 1e-5:
            still_timesteps += 1
        else:
            still_timesteps = 0
        cur_center = new_center
        cur_quat = new_quat
        wait_timesteps += 1

    if still_timesteps < 5000:
        printout(flog, 'Scene Object Not Still!\n')
        flog.close()
        env.close()
        exit(1)

    cur_center, cur_quat = env.get_scene_object_pose()
    cur_quat = Quaternion(cur_quat)
    start_angle_diff = Quaternion.absolute_distance(cur_quat, start_quat)/np.pi*180
    # still z-up
    if cur_quat.rotation_matrix[2, 2] < 0.999:
        printout(flog, '[FAIL] z is not up!\n')
        success = False
    # check motion len
    motion_dir = cur_center - start_center
    motion_len = np.linalg.norm(motion_dir)
    motion_dir /= motion_len
    if motion_len < 0.1:
        printout(flog, '[FAIL] not enough motion len (< 0.1)!\n')
        success = False
    # check motion dir
    if np.dot(motion_dir, cambase_xdir_world) < np.cos(np.pi/6):
        printout(flog, '[FAIL] too big motion direction diff (> 30 degree)!\n')
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

