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
parser.add_argument('--out_dir', type=str, default='./results/env_stacking')
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
# have a second camera which is in charge of generating data
cam2 = Camera(env, random_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
out_info['camera2_metadata'] = cam2.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# material
object_material = env.get_material(4, 4, 0.01)
env.set_default_material(object_material)
# acting object
acting_object_shape_id = np.random.choice(act_shapes)
acting_object_init_scale = get_random_number(0.3, 1.5)
env.load_acting_object(acting_object_shape_id, init_scale=acting_object_init_scale)
out_info['acting_object'] = acting_object_shape_id
out_info['acting_object_init_scale'] = acting_object_init_scale
env.render()

# simulate some steps for the scene object to fall down to the ground and stay rest
still_timesteps = 0
wait_timesteps = 0
cur_center, cur_quat = env.get_acting_object_pose()
cur_quat = Quaternion(cur_quat)
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    new_center, new_quat = env.get_acting_object_pose()
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
rgb, depth = cam2.get_observation()

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam2.compute_camera_XYZA(depth)
cam_XYZA = cam2.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

gt_applicable = np.zeros((cam2.image_size, cam2.image_size), dtype=np.bool)
gt_applicable[cam_XYZA_id1, cam_XYZA_id2] = True

# sample one pixel to interact
xs, ys = np.where(gt_applicable)
if len(xs) == 0:
    printout(flog, 'No Possible Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]
out_info['init_pixel_locs'] = [int(x), int(y)]

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam2.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]

# load scene object
init_x = position_world[0]
init_y = position_world[1]
init_z = position_world[2]
scene_object_setting = {'name': 'object', 'shape_id': args.shape_id, \
        'fix_root_link': False, 'state': 'locked', \
        'init_z': init_z, 'init_x': init_x, 'init_y': init_y,  \
        'init_quat': get_random_z_up_rot_quat()}
scene_object_setting = env.load_scene_object(scene_object_setting, is_putting=True)
out_info['scene_object'] = scene_object_setting
env.render()

# get link ids
scene_object_lids = [l.get_id() for l in env.scene_object.get_links()]
acting_object_lids = [l.get_id() for l in env.acting_object.get_links()]

# check: have no collision with the scene objects at beginning;
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
        if (aid1 in scene_object_lids) and (aid2 not in scene_object_lids):
            printout(flog, '[FAIL] collision at start!\n')
            flog.close()
            env.close()
            exit(1)
        if (aid2 in scene_object_lids) and (aid1 not in scene_object_lids):
            printout(flog, '[FAIL] collision at start!\n')
            flog.close()
            env.close()
            exit(1)

# drop it off
# both objects should stay at stable poses
still_timesteps = 0
wait_timesteps = 0
cur_scene_center, cur_scene_quat = env.get_scene_object_pose()
cur_scene_quat = Quaternion(cur_scene_quat)
cur_acting_center, cur_acting_quat = env.get_acting_object_pose()
cur_acting_quat = Quaternion(cur_acting_quat)
while still_timesteps < 5000 and wait_timesteps < 40000:
    env.step()
    env.render()
    new_scene_center, new_scene_quat = env.get_scene_object_pose()
    new_scene_quat = Quaternion(new_scene_quat)
    new_acting_center, new_acting_quat = env.get_acting_object_pose()
    new_acting_quat = Quaternion(new_acting_quat)
    scene_angle_diff = Quaternion.absolute_distance(cur_scene_quat, new_scene_quat)/np.pi*180
    acting_angle_diff = Quaternion.absolute_distance(cur_acting_quat, new_acting_quat)/np.pi*180
    if scene_angle_diff < 1 and acting_angle_diff < 1 and \
            np.max(np.abs(cur_scene_center - new_scene_center)) < 1e-4 and \
            np.max(np.abs(cur_acting_center - new_acting_center)) < 1e-4:
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_scene_center = new_scene_center
    cur_scene_quat = new_scene_quat
    cur_acting_center = new_acting_center
    cur_acting_quat = new_acting_quat
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, '[FAIL] scene or acting objects does not stay stably\n')
    flog.close()
    env.close()
    exit(1)

# the acting object must have z-up (so that only 1-dof rotation)
cur_acting_rotmat = env.acting_object.get_root_pose().to_transformation_matrix()
if cur_acting_rotmat[2, 2] < 1 - 1e-4:
    printout(flog, '[FAIL] the acting object must have z-up\n')
    flog.close()
    env.close()
    exit(1)

# there must be contact between scene-obj and act-obj
env.step()
succ = False
for c in env.scene.get_contacts():
    aid1 = c.actor1.get_id()
    aid2 = c.actor2.get_id()
    has_impulse = False
    for p in c.points:
        if abs(p.impulse @ p.impulse) > 1e-4:
            has_impulse = True
            break
    if has_impulse:
        if (aid1 in scene_object_lids) and (aid2 in acting_object_lids):
            succ = True
            break
        if (aid2 in scene_object_lids) and (aid1 in acting_object_lids):
            succ = True
            break
if not succ:
    printout(flog, '[FAIL] no contact between scene-obj and act-obj in the final scene state\n')
    flog.close()
    env.close()
    exit(1)

# re-create the scene (visible ground + scene-object)
env.show_ground()
cur_scene_object_pose = env.scene_object.get_root_pose()
cur_acting_object_pose = env.acting_object.get_root_pose()
cur_acting_object_pose = Pose(p=[cur_acting_object_pose.p[0] - cur_scene_object_pose.p[0], \
        cur_acting_object_pose.p[1] - cur_scene_object_pose.p[1], \
        cur_acting_object_pose.p[2]], q=cur_acting_object_pose.q)
cur_scene_object_pose = Pose(p=[0, 0, cur_scene_object_pose.p[2]], q=cur_scene_object_pose.q)
env.remove_scene_object()
env.remove_acting_object()
env.render()

# compute minimal grid-pixels distance
rgb, depth = cam.get_observation()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
min_dists_squared = ((cam_XYZA_pts[1:, :] - cam_XYZA_pts[0:1, :])**2).sum(axis=1).min()

# load scene object and re-render
env.load_object_with_pose(args.shape_id, cur_scene_object_pose, 'scene')
env.render()

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
gt_ground_valid = (cam.get_link_mask() == env.ground.get_id())
Image.fromarray(gt_ground_valid.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_applicable.png'))
Image.fromarray(gt_ground_valid.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_possible.png'))

# get the closed pixel xy to the succ dropping pixel
dropping_pt_world = np.array([cur_acting_object_pose.p[0], cur_acting_object_pose.p[1], env.ground_z], dtype=np.float32)
xs, ys = np.where(gt_ground_valid)
if len(xs) == 0:
    printout(flog, 'No Possible Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
positions_cam = cam_XYZA[xs, ys, :3]
positions_cam_xyz1 = np.ones((positions_cam.shape[0], 4), dtype=np.float32)
positions_cam_xyz1[:, :3] = positions_cam
positions_world_xyz1 = (cam.get_metadata()['mat44'] @ positions_cam_xyz1.T).T
positions_world = positions_world_xyz1[:, :3]
dists_squared = ((positions_world - np.expand_dims(dropping_pt_world, axis=0))**2).sum(axis=1)
dists_min_id = np.argmin(dists_squared)
if dists_squared[dists_min_id] < min_dists_squared:
    x = xs[dists_min_id]
    y = ys[dists_min_id]
else:
    printout(flog, '[FAIL] the succ dropping pixel is not visible\n')
    flog.close()
    env.close()
    exit(1)
out_info['pixel_locs'] = [int(x), int(y)]
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(out_dir, 'point_to_interact.png'))

# load acting object
env.load_object_with_pose(acting_object_shape_id, cur_acting_object_pose, 'acting', \
        init_scale=acting_object_init_scale)
env.render()

rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_start.png'))
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_final.png'))

# export final acting object collision mesh (cambase-space, zero-centered)
vs, fs = env.get_global_mesh(env.acting_object)
acting_object_center = env.acting_object.get_root_pose().p
vs[:, 0] -= acting_object_center[0]
vs[:, 1] -= acting_object_center[1]
vs[:, 2] -= acting_object_center[2]
acting_object_pts, _ = sample_points(vs, fs+1, num_points=1000)
acting_object_pts = acting_object_pts @ cam.get_metadata()['base_mat44'][:3, :3]
export_pts(os.path.join(out_dir, 'acting_object_cambase.pts'), acting_object_pts)

# remove acting object
env.remove_acting_object()
env.render()
rgb, depth = cam.get_observation()

# sample a neg position
idx = np.random.randint(len(xs))
while idx == dists_min_id:
    idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]
out_info['neg_pixel_locs'] = [int(x), int(y)]
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(out_dir, 'neg_point_to_interact.png'))

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]

# re-place acting object to a neg-position
cur_acting_object_pose = Pose(p=[position_world[0], position_world[1], \
        cur_acting_object_pose.p[2]], q=cur_acting_object_pose.q)
env.load_object_with_pose(acting_object_shape_id, cur_acting_object_pose, 'acting', \
        init_scale=acting_object_init_scale)
env.render()
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'neg_rgb_start.png'))

### wait to start
if not args.no_gui:
    ### wait to start
    env.wait_to_start()


### main steps
success = True

# drop it off
# check any motion
start_scene_center, start_scene_quat = env.get_scene_object_pose()
start_scene_quat = Quaternion(start_scene_quat)
start_acting_center, start_acting_quat = env.get_acting_object_pose()
start_acting_quat = Quaternion(start_acting_quat)
for _ in range(20000):
    env.step()
    env.render()
    cur_scene_center, cur_scene_quat = env.get_scene_object_pose()
    cur_scene_quat = Quaternion(cur_scene_quat)
    cur_acting_center, cur_acting_quat = env.get_acting_object_pose()
    cur_acting_quat = Quaternion(cur_acting_quat)
    scene_angle_diff = Quaternion.absolute_distance(start_scene_quat, cur_scene_quat)/np.pi*180
    acting_angle_diff = Quaternion.absolute_distance(start_acting_quat, cur_acting_quat)/np.pi*180
    if scene_angle_diff > 1 or acting_angle_diff > 1 or \
            np.max(np.abs(start_scene_center - cur_scene_center)) > 1e-4 or \
            np.max(np.abs(start_acting_center - cur_acting_center)) > 1e-4:
        success = False
        break

rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'neg_rgb_final.png'))

if not success:
    # THIS IS A NEG DATA
    # remove this pixel from possible
    gt_ground_valid[x, y] = False
    Image.fromarray(gt_ground_valid.astype(np.uint8)*255).save(os.path.join(out_dir, 'gt_possible.png'))

# always success
out_info['result'] = True
out_info['neg_result'] = success

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

