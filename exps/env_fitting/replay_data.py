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
scene_object_joint_angles = replay_data['scene_object_joint_angles']
env.load_scene_object(replay_data['scene_object'], joint_angles=scene_object_joint_angles)
z_max = replay_data['scene_object_z_max']

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
    print('Scene Object cannot remain at the starting articulated part poses!')
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
    print('Scene Object cannot be fully closed from the starting articulated part poses!')
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
    print('Scene Object cannot remain at the starting articulated part poses!')
    env.close()
    exit(1)


### use the GT vision
rgb, depth = cam.get_observation()

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

x, y = replay_data['pixel_locs'][0], replay_data['pixel_locs'][1]

# compute the intended dropping pixel link
id2link = dict()
for l in env.scene_object.get_links():
    id2link[l.get_id()] = l
link_mask = cam.get_link_mask()
dropping_pixel_link = id2link[link_mask[x, y]]
#print('dropping_pixel_link: ', dropping_pixel_link)

# get pixel 3D position (cam)
position_cam = cam_XYZA[x, y, :3]

# load acting object
env.load_acting_object(replay_data['acting_object'])

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
            print('[FAIL] collision at start!')
            success = False
            break
        if (aid2 in acting_object_lids) and (aid1 not in acting_object_lids):
            print('[FAIL] collision at start!')
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
            print('[FAIL] pose change or fall off table during falloff: first 2000 steps!')
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
            print('[FAIL] pose change or fall off table during falloff: final 10000 steps!')
            success = False
            break

# stitch the acting object to the dropping_link
# check: we can successfully close all articulated parts to the lowest limits
if success:
    env.stitch_acting_object(dropping_pixel_link, replay_data['position_world'])

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
        print('[FAIL] cannot close the articulated parts!')
        success = False

if success:
    print('[Successful Interaction] Done. Ctrl-C to quit.')
    while True:
        env.step()
        env.render()
else:
    print('[Unsuccessful Interaction] Failed. Ctrl-C to quit.')
    # close env
    env.close()

