"""
    Augmented Negative Data: gt_applicable - gt_possible
"""

import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
from pyquaternion import Quaternion
from camera import Camera
from geometry_utils import load_pts


class SAPIENVisionDataset(data.Dataset):

    def __init__(self, category_types, data_features, \
            env_name=None, buffer_max_num=None, img_size=224, \
            no_true_false_equal=False, no_aug_neg_data=False, only_true_data=False):
        self.category_types = category_types

        self.env_name = env_name
        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        
        self.no_true_false_equal = no_true_false_equal
        self.no_aug_neg_data = no_aug_neg_data
        self.only_true_data = only_true_data

        # data buffer
        self.true_data = []
        self.false_data = []

        # data features
        self.data_features = data_features
        
    def load_data(self, data_list):
        bar = ProgressBar()
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            cur_shape_id, cur_category, cur_epoch_id, cur_trial_id  = cur_dir.split('/')[-1].split('_')

            if (self.category_types is not None) and (cur_category not in self.category_types):
                continue

            with open(os.path.join(cur_dir, 'result.json'), 'r') as fin:
                result_data = json.load(fin)

                ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)
                
                success = result_data['result']
                cam2cambase = np.array(result_data['camera_metadata']['cam2cambase'], dtype=np.float32)

                # load original data
                if success:
                    cur_data = (cur_dir, cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                            ori_pixel_ids, pixel_ids, True, True, cam2cambase)
                    self.true_data.append(cur_data)
                else:
                    if not self.only_true_data:
                        cur_data = (cur_dir, cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, True, False, cam2cambase)
                        self.false_data.append(cur_data)
                
                # load augmented false data
                if not self.no_aug_neg_data:
                    aug_pixels_map = np.zeros((448, 448), dtype=np.bool)
                    with Image.open(os.path.join(cur_dir, 'gt_applicable.png')) as fimg:
                        aug_pixels_map[np.array(fimg) > 128] = True
                    with Image.open(os.path.join(cur_dir, 'gt_possible.png')) as fimg:
                        aug_pixels_map[np.array(fimg) > 128] = False
                    xs, ys = np.where(aug_pixels_map)
                    if len(xs) > 0:
                        idx = np.random.randint(len(xs))
                        x, y = xs[idx], ys[idx]
                        ori_pixel_ids = np.array([int(x), int(y)], dtype=np.int32)
                        pixel_ids = np.round(np.array([int(x), int(y)], dtype=np.float32) / 448 * self.img_size).astype(np.int32)
                        cur_data = (cur_dir, cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, False, False, cam2cambase)
                        self.false_data.append(cur_data)

        # delete data if buffer full
        if self.buffer_max_num is not None:
            if len(self.true_data) > self.buffer_max_num:
                self.true_data = self.true_data[-self.buffer_max_num:]
            if len(self.false_data) > self.buffer_max_num:
                self.false_data = self.false_data[-self.buffer_max_num:]

    def __str__(self):
        strout = '[SAPIENVisionDataset %s %d] img_size: %d, no_aug_neg_data: %s\n' % (self.env_name, len(self), self.img_size, 'True' if self.no_aug_neg_data else 'False')
        strout += '\tTrue: %d False: %d\n' % (len(self.true_data), len(self.false_data))
        return strout

    def __len__(self):
        if self.no_true_false_equal:
            return len(self.true_data) + len(self.false_data)
        else:
            return max(len(self.true_data), len(self.false_data)) * 2

    def __getitem__(self, index):
        if self.no_true_false_equal:
            if index < len(self.false_data):
                cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
                        is_original, result, cam2cambase = \
                            self.false_data[index]
            else:
                cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
                        is_original, result, cam2cambase = \
                            self.true_data[index - len(self.false_data)]
        else:
            if index % 2 == 0:
                cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
                        is_original, result, cam2cambase = \
                            self.false_data[(index//2) % len(self.false_data)]
            else:
                cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
                        is_original, result, cam2cambase = \
                            self.true_data[(index//2) % len(self.true_data)]

        # pre-load some data
        if any(feat in ['gt_applicable_img', 'gt_applicable_pc'] for feat in self.data_features):
            with Image.open(os.path.join(cur_dir, 'gt_applicable.png')) as fimg:
                gt_applicable_img = np.array(fimg, dtype=np.float32) > 128

        if any(feat in ['gt_possible_img', 'gt_possible_pc'] for feat in self.data_features):
            with Image.open(os.path.join(cur_dir, 'gt_possible.png')) as fimg:
                gt_possible_img = np.array(fimg, dtype=np.float32) > 128

        if any(feat in ['scene_pc_cam', 'scene_pc_pxids', 'gt_applicable_pc', 'gt_possible_pc'] for feat in self.data_features):
            x, y = ori_pixel_ids[0], ori_pixel_ids[1]
            with h5py.File(os.path.join(cur_dir, 'cam_XYZA.h5'), 'r') as fin:
                cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                cam_XYZA_pts = fin['pc'][:].astype(np.float32)
            scene_pc_img = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
            
            pt = scene_pc_img[x, y, :3]
            if 'scene_pc_pxids' in self.data_features:
                pc_ptid = np.array([x, y], dtype=np.int32)
            if 'gt_applicable_pc' in self.data_features:
                gt_applicable_pt = gt_applicable_img[x, y]
            if 'gt_possible_pc' in self.data_features:
                gt_possible_pt = gt_possible_img[x, y]
            
            mask = (scene_pc_img[:, :, 3] > 0.5)
            mask[x, y] = False
            scene_pc_cam = scene_pc_img[mask, :3]
            if 'scene_pc_pxids' in self.data_features:
                grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
                grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
                pc_pxids = grid_xy[:, mask].T
            if 'gt_applicable_pc' in self.data_features:
                gt_applicable_pc = gt_applicable_img[mask]
            if 'gt_possible_pc' in self.data_features:
                gt_possible_pc = gt_possible_img[mask]
            
            idx = np.arange(scene_pc_cam.shape[0])
            np.random.shuffle(idx)
            while len(idx) < 30000:
                idx = np.concatenate([idx, idx])
            idx = idx[:30000-1]
            scene_pc_cam = scene_pc_cam[idx, :]
            scene_pc_cam = np.vstack([pt, scene_pc_cam])
            if 'scene_pc_pxids' in self.data_features:
                pc_pxids = np.vstack([pc_ptid, pc_pxids[idx, :]])
            if 'gt_applicable_pc' in self.data_features:
                gt_applicable_pc = np.append(gt_applicable_pt, gt_applicable_pc[idx])
            if 'gt_possible_pc' in self.data_features:
                gt_possible_pc = np.append(gt_possible_pt, gt_possible_pc[idx])
        
        # output all require features
        data_feats = ()
        for feat in self.data_features:
            if feat == 'rgb':
                with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'normal':
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with Image.open(os.path.join(cur_dir, 'gt_nor.png')) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                out = out[x, y, :3] * 2 - 1
                out = torch.from_numpy(out).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'rgb_start':
                if is_original:
                    with Image.open(os.path.join(cur_dir, 'rgb_start.png')) as fimg:
                        out = np.array(fimg, dtype=np.float32) / 255
                    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                else:
                    out = torch.ones(1, 3, 448, 448).float()
                data_feats = data_feats + (out,)
             
            elif feat == 'rgb_final':
                if is_original:
                    with Image.open(os.path.join(cur_dir, 'rgb_final.png')) as fimg:
                        out = np.array(fimg, dtype=np.float32) / 255
                    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                else:
                    out = torch.ones(1, 3, 448, 448).float()
                data_feats = data_feats + (out,)
            
            elif feat == 'rgb_point':
                if is_original:
                    with Image.open(os.path.join(cur_dir, 'point_to_interact.png')) as fimg:
                        out = np.array(fimg, dtype=np.float32) / 255
                    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                else:
                    out = torch.ones(1, 3, 448, 448).float()
                data_feats = data_feats + (out,)

            elif feat == 'gt_applicable_img':
                out = torch.from_numpy(gt_applicable_img).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'gt_applicable_pc':
                out = torch.from_numpy(gt_applicable_pc).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gt_possible_img':
                out = torch.from_numpy(gt_possible_img).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'gt_possible_pc':
                out = torch.from_numpy(gt_possible_pc).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gt_possible_img':
                out = torch.from_numpy(gt_possible_img).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'scene_pc_cam':
                out = torch.from_numpy(scene_pc_cam).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'scene_pc_pxids':
                out = torch.from_numpy(pc_pxids).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'acting_pc_cambase':
                pc = load_pts(os.path.join(cur_dir, 'acting_object_cambase.pts'))
                if self.env_name is not None:
                    pc_z_size = pc[:, 2].max() - pc[:, 2].min()
                    pc[:, 2] += pc_z_size / 2
                    if self.env_name in ['pushing', 'rotating']:
                        pc_x_size = pc[:, 0].max() - pc[:, 0].min()
                        pc[:, 0] -= pc_x_size / 2
                out = torch.from_numpy(pc).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'cam2cambase':
                out = np.array(cam2cambase, dtype=np.float32)
                out = torch.from_numpy(out).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'is_original':
                data_feats = data_feats + (is_original,)
            
            elif feat == 'pixel_id':
                out = torch.from_numpy(pixel_ids).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'result':
                data_feats = data_feats + (result,)

            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'category':
                data_feats = data_feats + (category,)

            elif feat == 'epoch_id':
                data_feats = data_feats + (epoch_id,)

            elif feat == 'trial_id':
                data_feats = data_feats + (trial_id,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

