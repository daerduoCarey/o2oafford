"""
    Stacking
"""

from __future__ import division
import os
import sys
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
import copy
import json

from utils import process_angle_limit, get_random_number, printout


class ContactError(Exception):
    pass


class Env(object):
    
    def __init__(self, flog=None, show_gui=True, \
            render_rate=20, timestep=1/500, succ_ratio=0.1, delete_thres=-2):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.delete_thres = delete_thres

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True
        
        self.renderer = sapien.OptifuserRenderer(config=render_config)
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1])

        # default Nones
        self.scene_object = None
        self.acting_object = None

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def get_global_mesh(self, obj):
        final_vs = []; final_fs = []; vid = 0;
        for l in obj.get_links():
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0]; v[:, 1] *= vscale[1]; v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs
     
    def load_object_with_pose(self, shape_id, pose, name, urdf_fn=None, material=None, init_scale=1.0):
        if material is None: material = self.default_material

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = init_scale
        if shape_id is not None:
            urdf_fn = '../data/sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
        new_obj =  loader.load(urdf_fn, \
                {"material": material})
        new_obj.set_root_pose(pose)
        if shape_id is None:
            printout(self.flog, '[env::load_object_with_pose] loaded: %s' % urdf_fn)
        else:
            printout(self.flog, '[env::load_object_with_pose] loaded: %s' % shape_id)

        # disable part articulation
        if shape_id is not None:
            with open('../stats/stable_actobjs_qpos.json', 'r') as fin:
                locked_qpos = json.load(fin)[shape_id]
            if len(locked_qpos) > 0:
                jid = 0
                for j in new_obj.get_joints():
                    if j.get_dof() == 1:
                        j.set_limits(np.array([[locked_qpos[jid], locked_qpos[jid]]], dtype=np.float32))
                        jid += 1
                new_obj.set_qpos(np.array(locked_qpos, dtype=np.float32))

        if name == 'scene':
            self.scene_object = new_obj
        elif name == 'acting':
            self.acting_object = new_obj

    def load_acting_object(self, shape_id, urdf_fn=None, material=None, init_scale=1.0):
        if material is None: material = self.default_material

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = init_scale
        if shape_id is not None:
            urdf_fn =  '../data/sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
        self.acting_object = loader.load(urdf_fn, \
                {"material": material})
        pose = Pose([0, 0, 0], [0.5, 0, 0, 0.8])
        self.acting_object.set_root_pose(pose)
        if shape_id is None:
            printout(self.flog, '[env::load_acting_object] loaded: %s' % urdf_fn)
        else:
            printout(self.flog, '[env::load_acting_object] loaded: %s' % shape_id)

        # disable part articulation
        if shape_id is not None:
            with open('../stats/stable_actobjs_qpos.json', 'r') as fin:
                locked_qpos = json.load(fin)[shape_id]
            if len(locked_qpos) > 0:
                jid = 0
                for j in self.acting_object.get_joints():
                    if j.get_dof() == 1:
                        j.set_limits(np.array([[locked_qpos[jid], locked_qpos[jid]]], dtype=np.float32))
                        jid += 1
                self.acting_object.set_qpos(np.array(locked_qpos, dtype=np.float32))

        # add an invisible ground
        final_vs, _ = self.get_global_mesh(self.acting_object)
        z_min = np.min(final_vs[:, 2])
        self.ground = self.scene.add_ground(z_min-0.01, render=False)
        self.ground_z = z_min-0.01
    
    def load_scene_object(self, item, material=None, is_putting=False):
        if material is None: material = self.default_material

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        if 'shape_id' in item:
            item['urdf_fn'] = '../data/sapien_dataset/%s/mobility_vhacd.urdf' % item['shape_id']
        self.scene_object = loader.load(item['urdf_fn'], \
                {"material": material})
        if 'init_x' not in item: item['init_x'] = 0
        if 'init_y' not in item: item['init_y'] = 0
        if 'init_z' not in item: item['init_z'] = 0
        if 'init_quat' not in item: item['init_quat'] = [1, 0, 0, 0]
        if item['state'] == 'locked':
            if 'shape_id' in item:
                with open('../stats/stable_actobjs_qpos.json', 'r') as fin:
                    locked_qpos = json.load(fin)[item['shape_id']]
                if len(locked_qpos) > 0:
                    jid = 0
                    for j in self.scene_object.get_joints():
                        if j.get_dof() == 1:
                            j.set_limits(np.array([[locked_qpos[jid], locked_qpos[jid]]], dtype=np.float32))
                            jid += 1
        if 'joint_angles' not in item:
            joint_angles = []
            for j in self.scene_object.get_joints():
                if j.get_dof() == 1:
                    l = process_angle_limit(j.get_limits()[0, 0])
                    r = process_angle_limit(j.get_limits()[0, 1])
                    if item['state'] == 'closed':
                        joint_angles.append(float(l))
                    elif item['state'] == 'locked':
                        joint_angles.append(float(l))
                    elif item['state'] == 'open':
                        joint_angles.append(float(r))
                    elif item['state'] == 'random-middle':
                        joint_angles.append(float(get_random_number(l, r)))
                    elif item['state'] == 'random-closed-middle':
                        if np.random.random() < 0.5:
                            joint_angles.append(float(get_random_number(l, r)))
                        else:
                            joint_angles.append(float(l))
                    else:
                        raise ValueError('ERROR: object init state %s unknown!' % state)
            item['joint_angles'] = joint_angles
        self.scene_object.set_qpos(item['joint_angles'])
        pose = Pose([item['init_x'], item['init_y'], item['init_z']], item['init_quat'])
        self.scene_object.set_root_pose(pose)
        printout(self.flog, '[env::load_scene_object] loaded: %s' % str(item))

        if is_putting:
            final_vs, _ = self.get_global_mesh(self.scene_object)
            z_min = np.min(final_vs[:, 2])
            item['init_z'] += item['init_z'] - z_min + 2
            pose = Pose([item['init_x'], item['init_y'], item['init_z']], item['init_quat'])
            self.scene_object.set_root_pose(pose)

        return item

    def remove_scene_object(self):
        self.scene.remove_articulation(self.scene_object)
        self.scene_object = None

    def remove_acting_object(self):
        self.scene.remove_articulation(self.acting_object)
        self.acting_object = None

    def show_ground(self):
        self.scene.remove_actor(self.ground)
        self.ground = self.scene.add_ground(self.ground_z)
     
    def get_acting_object_pose(self):
        pose = self.acting_object.get_root_pose()
        return pose.p, pose.q

    def get_scene_object_pose(self):
        pose = self.scene_object.get_root_pose()
        return pose.p, pose.q
        
    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def set_default_material(self, material):
        self.default_material = material

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        self.scene.step()

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None


