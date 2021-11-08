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
env.load_scene_object(replay_data['scene_object'])

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
    print('Scene Object Not Still!')
    env.close()
    exit(1)


### use the GT vision
rgb, depth = cam.get_observation()

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

x, y = replay_data['pixel_locs'][0], replay_data['pixel_locs'][1]

# get pixel 3D position (cam)
position_cam = cam_XYZA[x, y, :3]

# compute cambase front/x-direction
cambase_xdir_world = cam.get_metadata()['base_mat44'][:3, 0]

# load acting object
env.load_acting_object(replay_data['acting_object'])

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
                print('[FAIL] collision at start!')
                success = False
                break
            if (aid2 in acting_object_lids) and (aid1 not in acting_object_lids):
                print('[FAIL] collision at start!')
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
        print('Scene Object Not Still!')
        env.close()
        exit(1)

    cur_center, cur_quat = env.get_scene_object_pose()
    cur_quat = Quaternion(cur_quat)
    start_angle_diff = Quaternion.absolute_distance(cur_quat, start_quat)/np.pi*180
    # still z-up
    if cur_quat.rotation_matrix[2, 2] < 0.999:
        print('[FAIL] z is not up!')
        success = False
    # check motion len
    motion_dir = cur_center - start_center
    motion_len = np.linalg.norm(motion_dir)
    motion_dir /= motion_len
    if motion_len < 0.1:
        print('[FAIL] not enough motion len (< 0.1)!')
        success = False
    # check motion dir
    if np.dot(motion_dir, cambase_xdir_world) < np.cos(np.pi/6):
        print('[FAIL] too big motion direction diff (> 30 degree)!')
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

