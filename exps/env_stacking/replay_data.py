"""
    Replay
"""

import os
import sys
import shutil
import numpy as np
import json
import h5py
from pyquaternion import Quaternion

from sapien.core import Pose
from env import Env, ContactError

from robots.panda_robot import Robot
from camera import Camera
from utils import get_global_position_from_camera

out_dir = sys.argv[1]
json_fn = os.path.join(out_dir, 'result.json')
with open(json_fn, 'r') as fin:
    replay_data = json.load(fin)
    print(replay_data)

# setup env
env = Env()

# setup camera
cam_theta = replay_data['camera_metadata']['theta']
cam_phi = replay_data['camera_metadata']['phi']
cam = Camera(env, theta=cam_theta, phi=cam_phi)
env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# load scene objects
object_material = env.get_material(4, 4, 0.01)
env.set_default_material(object_material)

env.load_acting_object(replay_data['acting_object'], init_scale=replay_data['acting_object_init_scale'])
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
    print('Scene Object Not Still!')
    env.close()
    exit(1)

# load scene object
env.load_scene_object(replay_data['scene_object'])

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
            print('[FAIL] collision at start!')
            env.close()
            exit(1)
        if (aid2 in scene_object_lids) and (aid1 not in scene_object_lids):
            print('[FAIL] collision at start!')
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
    print('[FAIL] scene or acting objects does not stay stably')
    env.close()
    exit(1)

# the acting object must have z-up (so that only 1-dof rotation)
cur_acting_rotmat = env.acting_object.get_root_pose().to_transformation_matrix()
if cur_acting_rotmat[2, 2] < 1 - 1e-4:
    print('[FAIL] the acting object must have z-up')
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
    print('[FAIL] no contact between scene-obj and act-obj in the final scene state')
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
env.load_object_with_pose(replay_data['scene_object']['shape_id'], cur_scene_object_pose, 'scene')
env.render()

### use the GT vision
rgb, depth = cam.get_observation()

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
#export_pts(os.path.join(out_dir, 'scene_cambase.pts'), cam_XYZA_pts @ cam.get_metadata()['cam2cambase'][:3, :3].T)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

gt_ground_valid = (cam.get_link_mask() == env.ground.get_id())

# get the closed pixel xy to the succ dropping pixel
dropping_pt_world = np.array([cur_acting_object_pose.p[0], cur_acting_object_pose.p[1], env.ground_z], dtype=np.float32)
xs, ys = np.where(gt_ground_valid)
if len(xs) == 0:
    print('No Possible Pixel! Quit!')
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
    print('[FAIL] the succ dropping pixel is not visible')
    env.close()
    exit(1)

# load acting object
env.load_object_with_pose(replay_data['acting_object'], cur_acting_object_pose, 'acting', \
        init_scale=replay_data['acting_object_init_scale'])

# wait to show the viz
for _ in range(5000):
    env.step()
    env.render()

# remove acting object
env.remove_acting_object()
x, y = replay_data['neg_pixel_locs'][0], replay_data['neg_pixel_locs'][1]

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]

# re-place acting object to a neg-position
cur_acting_object_pose = Pose(p=[position_world[0], position_world[1], \
        cur_acting_object_pose.p[2]], q=cur_acting_object_pose.q)
env.load_object_with_pose(replay_data['acting_object'], cur_acting_object_pose, 'acting', \
        init_scale=replay_data['acting_object_init_scale'])
env.render()

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
        #break

if success:
    print('[Successful Interaction] Done. Ctrl-C to quit.')
    while True:
        env.step()
        env.render()
else:
    print('[Unsuccessful Interaction] Failed. Ctrl-C to quit.')
    # close env
    env.close()

