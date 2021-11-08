import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils
from subprocess import call
from progressbar import ProgressBar
from sklearn.metrics import average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from data import SAPIENVisionDataset
from datagen import DataGen

from pointnet2_ops.pointnet2_utils import furthest_point_sample


# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--data_dir', type=str, help='test data data dir')
parser.add_argument('--model_version', type=str, help='model version')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--model_epoch', type=int, help='ckpt epoch [if None, pick the last one]', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--visu_cnt_per_page', type=int, default=10)
parser.add_argument('--visu_batch', type=int, default=1)
parser.add_argument('--true_thres', type=float, default=0.5)
parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')
parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
parser.add_argument('--show_failure', action='store_true', default=False, help='only show failure cases [default: False]')
parser.add_argument('--actobj_random_size_rotation', action='store_true', default=False, help='only show failure cases [default: False]')
eval_conf = parser.parse_args()

# load train config
train_conf = torch.load(os.path.join('logs', eval_conf.exp_name, 'conf.pth'))

# load model
model_def = utils.get_model_module(eval_conf.model_version)

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')

# pick the last epoch is not specified
if eval_conf.model_epoch is None:
    eval_conf.model_epoch = -1
    ckpt_dir = os.path.join('logs', eval_conf.exp_name, 'ckpts')
    for item in os.listdir(ckpt_dir):
        if '-network.pth' in item:
            cur_epoch = int(item.split('-')[0])
            if cur_epoch > eval_conf.model_epoch:
                eval_conf.model_epoch = cur_epoch

# check if eval results already exist. If so, delete it.
result_dir = os.path.join('logs', eval_conf.exp_name, f'test-whole-model_epoch_{eval_conf.model_epoch}-succ_thres_{int(eval_conf.true_thres*100)}-{eval_conf.result_suffix}')
if eval_conf.actobj_random_size_rotation:
    result_dir = os.path.join('logs', eval_conf.exp_name, f'test-whole-actobj_random_size_rotation-model_epoch_{eval_conf.model_epoch}-succ_thres_{int(eval_conf.true_thres*100)}-{eval_conf.result_suffix}')
if eval_conf.show_failure:
    result_dir = os.path.join('logs', eval_conf.exp_name, f'test-whole-show_failure-model_epoch_{eval_conf.model_epoch}-succ_thres_{int(eval_conf.true_thres*100)}-{eval_conf.result_suffix}')
if os.path.exists(result_dir):
    if not eval_conf.overwrite:
        response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
        if response != 'y':
            sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# output result dir
out_dir = os.path.join(result_dir, 'out')
os.mkdir(out_dir)

# visu dir
if not eval_conf.no_visu:
    visu_dir = os.path.join(result_dir, 'visu')
    os.mkdir(visu_dir)
    input_scene_pc_cam_dir = os.path.join(visu_dir, 'input_scene_pc_cam')
    input_scene_pc_cambase_dir = os.path.join(visu_dir, 'input_scene_pc_cambase')
    input_acting_pc_cam_dir = os.path.join(visu_dir, 'input_acting_pc_cam')
    input_acting_pc_cambase_dir = os.path.join(visu_dir, 'input_acting_pc_cambase')
    pred_all_result_dir = os.path.join(visu_dir, 'pred_all_result')
    viz_rgb_dir = os.path.join(visu_dir, 'viz_rgb')
    gt_applicable_dir = os.path.join(visu_dir, 'gt_applicable')
    gt_possible_dir = os.path.join(visu_dir, 'gt_possible')
    viz_rgb_point_dir = os.path.join(visu_dir, 'viz_rgb_point')
    viz_rgb_start_dir = os.path.join(visu_dir, 'viz_rgb_start')
    viz_rgb_final_dir = os.path.join(visu_dir, 'viz_rgb_final')
    info_dir = os.path.join(visu_dir, 'info')
    os.mkdir(input_scene_pc_cam_dir)
    os.mkdir(input_scene_pc_cambase_dir)
    os.mkdir(input_acting_pc_cam_dir)
    os.mkdir(input_acting_pc_cambase_dir)
    os.mkdir(pred_all_result_dir)
    os.mkdir(viz_rgb_dir)
    os.mkdir(gt_applicable_dir)
    os.mkdir(gt_possible_dir)
    os.mkdir(viz_rgb_point_dir)
    os.mkdir(viz_rgb_start_dir)
    os.mkdir(viz_rgb_final_dir)
    os.mkdir(info_dir)

# dataset
data_features = ['scene_pc_cam', 'cam2cambase', 'acting_pc_cambase', \
        'rgb', 'rgb_point', 'rgb_start', 'rgb_final', \
        'gt_applicable_img', 'gt_possible_img', 'gt_applicable_pc', \
        'result', 'cur_dir', 'shape_id']

with open(os.path.join(eval_conf.data_dir, 'data_tuple_list.txt'), 'r') as fin:
    data_list = [os.path.join(eval_conf.data_dir, l.rstrip()) for l in fin.readlines()]
print('len(data_list): %d' % len(data_list))


# parse params
category_types = None
if train_conf.category_types is not None:
    category_types = train_conf.category_types.split(',')

dataset = SAPIENVisionDataset(category_types, data_features, \
        env_name=train_conf.env_name, img_size=train_conf.img_size, \
        no_true_false_equal=eval_conf.no_true_false_equal, \
        no_aug_neg_data=True, only_true_data=True)
dataset.load_data(data_list)
print(dataset)

if eval_conf.actobj_random_size_rotation:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=False, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
else:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=eval_conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=False, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)

# create models
network = model_def.Network(train_conf.feat_dim)

# load pretrained model
print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)
data_to_restore = torch.load(os.path.join('logs', eval_conf.exp_name, 'ckpts', '%d-network.pth' % eval_conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send to device
network.to(device)

# set models to evaluation mode
network.eval()

# compute numbers
tot_true = 0
tot_true_correct = 0
tot_false = 0
tot_false_correct = 0
gt_labels = np.zeros(0).astype(np.float32)
pred_labels = np.zeros(0).astype(np.float32)

# test over all data
with torch.no_grad():
    t = 0
    for batch_id, batch in enumerate(dataloader, 0):
        print('[%d/%d] testing....' % (t, len(dataset)))

        # prepare input
        input_scene_pcs_cam = torch.cat(batch[data_features.index('scene_pc_cam')], dim=0).to(eval_conf.device)  # B x 3N x 3
        gt_applicable_pc = torch.cat(batch[data_features.index('gt_applicable_pc')], dim=0).to(eval_conf.device) # B x 3N
        batch_size = input_scene_pcs_cam.shape[0]
        # fps to 10K-points
        input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, train_conf.num_point_per_shape).long().reshape(-1)  # BN
        input_pcid2 = furthest_point_sample(input_scene_pcs_cam, train_conf.num_point_per_shape).long().reshape(-1)           # BN
        input_scene_pcs_cam = input_scene_pcs_cam[input_pcid1, input_pcid2, :].reshape(batch_size, train_conf.num_point_per_shape, -1)    # B x N x 3
        gt_applicable_pc = gt_applicable_pc[input_pcid1, input_pcid2].reshape(batch_size, train_conf.num_point_per_shape)    # B x N
        
        # convert to cambase (z-up), normalize to zero-center
        cam2cambase_rotmats = torch.cat(batch[data_features.index('cam2cambase')], dim=0).to(eval_conf.device)      # B x 3 x 3
        input_scene_pcs_cambase = torch.matmul(input_scene_pcs_cam, cam2cambase_rotmats.permute(0, 2, 1))
        pc_centers = (input_scene_pcs_cambase.max(dim=1, keepdim=True)[0] + input_scene_pcs_cambase.min(dim=1, keepdim=True)[0]) / 2
        input_scene_pcs_cambase -= pc_centers

        input_acting_pcs_cambase = torch.cat(batch[data_features.index('acting_pc_cambase')], dim=0).to(eval_conf.device)    # B x N' x 3
        input_acting_pcs_cam = torch.matmul(input_acting_pcs_cambase, cam2cambase_rotmats)

        # augment the actobj_random_size_rotation
        if eval_conf.actobj_random_size_rotation:
            input_scene_pcs_cambase = input_scene_pcs_cambase.repeat(eval_conf.batch_size, 1, 1)
            input_scene_pcs_cam = input_scene_pcs_cam.repeat(eval_conf.batch_size, 1, 1)
            gt_applicable_pc = gt_applicable_pc.repeat(eval_conf.batch_size, 1)
            input_acting_pcs_cambase = input_acting_pcs_cambase.repeat(eval_conf.batch_size, 1, 1)
            rand_rotmats = utils.get_random_z_up_rot_pytorch_matrix(eval_conf.batch_size).to(input_acting_pcs_cambase.device)
            rand_sizes = torch.rand(eval_conf.batch_size).to(input_acting_pcs_cambase.device).unsqueeze(-1).unsqueeze(-1)
            input_acting_pcs_cambase = torch.matmul(input_acting_pcs_cambase, rand_rotmats) * rand_sizes
            input_acting_pcs_cam = torch.matmul(input_acting_pcs_cambase, cam2cambase_rotmats.repeat(eval_conf.batch_size, 1, 1))

        # forward through the network
        pred_result_logits, end_points = network(input_scene_pcs_cambase, input_acting_pcs_cambase)     # B

        # inference all results
        pred_all_result = network.inference_whole_pc(input_scene_pcs_cambase, input_acting_pcs_cambase) # B x N
        pred_all_result[~gt_applicable_pc] = 0
        
        # test numbers
        if not eval_conf.actobj_random_size_rotation:
            # prepare gt
            gt_result = torch.Tensor(batch[data_features.index('result')]).long().to(eval_conf.device)     # B
    
            # for each type of loss, compute losses per data
            result_loss_per_data = network.critic.get_ce_loss(pred_result_logits, gt_result)
        
            # compute true-score
            pred_result_true_score = torch.sigmoid(pred_result_logits)

            # for ap
            gt_labels = np.append(gt_labels, gt_result.cpu().numpy())
            pred_labels = np.append(pred_labels, pred_result_true_score.cpu().numpy())

            # compute numbers
            for i in range(batch_size):
                cur_pred_true_score = pred_result_true_score[i].item()
                cur_gt_result = gt_result[i].item()
                if cur_gt_result > 0.5:
                    tot_true += 1
                    if cur_pred_true_score > eval_conf.true_thres:
                        tot_true_correct += 1
                else:
                    tot_false += 1
                    if cur_pred_true_score < eval_conf.true_thres:
                        tot_false_correct += 1
            
        # prepare viz
        viz_rgb = torch.cat(batch[data_features.index('rgb')], dim=0).to(eval_conf.device)     # B x 3 x H x W
        gt_applicable_img = torch.cat(batch[data_features.index('gt_applicable_img')], dim=0).to(eval_conf.device) # B x H x W
        gt_possible_img = torch.cat(batch[data_features.index('gt_possible_img')], dim=0).to(eval_conf.device)     # B x H x W
        viz_rgb_point = torch.cat(batch[data_features.index('rgb_point')], dim=0).to(eval_conf.device)     # B x 3 x H x W
        viz_rgb_start = torch.cat(batch[data_features.index('rgb_start')], dim=0).to(eval_conf.device)     # B x 3 x H x W
        viz_rgb_final = torch.cat(batch[data_features.index('rgb_final')], dim=0).to(eval_conf.device)     # B x 3 x H x W

        if eval_conf.actobj_random_size_rotation:
            viz_rgb = viz_rgb.repeat(eval_conf.batch_size, 1, 1, 1)
   
        # visu
        if (not eval_conf.no_visu) and batch_id < eval_conf.visu_batch:
            print('Visualizing ...')
            viz_batch_size = batch_size
            if eval_conf.actobj_random_size_rotation:
                viz_batch_size = eval_conf.batch_size
            for i in range(viz_batch_size):
                if (not eval_conf.show_failure) or ((pred_result_true_score[i].item() > eval_conf.true_thres) != (gt_result[i].item() > 0.5)):
                    fn = 'data-%03d.png' % (batch_id * viz_batch_size + i)
                    utils.render_pts(os.path.join(BASE_DIR, input_scene_pc_cam_dir, fn.split('.')[0]), input_scene_pcs_cam[i].cpu().numpy(), highlight_id=0, campos=0)
                    utils.render_pts(os.path.join(BASE_DIR, input_scene_pc_cambase_dir, fn.split('.')[0]), input_scene_pcs_cambase[i].cpu().numpy(), highlight_id=0)
                    utils.render_pts(os.path.join(BASE_DIR, input_acting_pc_cam_dir, fn.split('.')[0]), input_acting_pcs_cam[i].cpu().numpy())
                    utils.render_pts(os.path.join(BASE_DIR, input_acting_pc_cambase_dir, fn.split('.')[0]), input_acting_pcs_cambase[i].cpu().numpy())
                    utils.render_pts_label_png(os.path.join(BASE_DIR, pred_all_result_dir, fn.split('.')[0]), input_scene_pcs_cam[i].cpu().numpy(), pred_all_result[i].cpu().numpy(), campos=0)
                    img_toshow = (viz_rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_toshow).save(os.path.join(viz_rgb_dir, fn))
                    if not eval_conf.actobj_random_size_rotation:
                        img_toshow = gt_applicable_img[i].cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(img_toshow).save(os.path.join(gt_applicable_dir, fn))
                        img_toshow = gt_possible_img[i].cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(img_toshow).save(os.path.join(gt_possible_dir, fn))
                        img_toshow = (viz_rgb_point[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_toshow).save(os.path.join(viz_rgb_point_dir, fn))
                        img_toshow = (viz_rgb_start[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_toshow).save(os.path.join(viz_rgb_start_dir, fn))
                        img_toshow = (viz_rgb_final[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_toshow).save(os.path.join(viz_rgb_final_dir, fn))
                    with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                        if eval_conf.actobj_random_size_rotation:
                            fout.write('cur_dir: %s\n' % batch[data_features.index('cur_dir')][0])
                        else:
                            fout.write('cur_dir: %s\n' % batch[data_features.index('cur_dir')][i])
                            fout.write('pred: %s\n' % utils.print_true_false((pred_result_logits[i]>0).cpu().numpy()))
                            fout.write('gt: %s\n' % utils.print_true_false(gt_result[i].cpu().numpy()))
                            fout.write('result_loss: %f\n' % result_loss_per_data[i].item())

            # visu a html
            if batch_id == eval_conf.visu_batch - 1:
                print('Generating html visualization ...')
                if eval_conf.actobj_random_size_rotation:
                    sublist = 'input_scene_pc_cambase,input_acting_pc_cambase,input_scene_pc_cam,input_acting_pc_cam,pred_all_result,viz_rgb,info'
                else:
                    sublist = 'input_scene_pc_cambase,input_acting_pc_cambase,input_scene_pc_cam,input_acting_pc_cam,pred_all_result,viz_rgb,gt_applicable,gt_possible,viz_rgb_point,viz_rgb_start,viz_rgb_final,info'
                cmd = 'cd %s && python %s . %d htmls %s %s > /dev/null' % (visu_dir, os.path.join(BASE_DIR, 'gen_html_hierachy_local.py'), eval_conf.visu_cnt_per_page, sublist, sublist)
                call(cmd, shell=True)
                print('DONE')
                exit(1)
                if eval_conf.actobj_random_size_rotation:
                    break
        
        t += batch_size

if not eval_conf.actobj_random_size_rotation:
    with open(os.path.join(result_dir, 'results.txt'), 'w') as fout:
        tp = tot_true_correct
        fn = tot_true - tot_true_correct
        tn = tot_false_correct
        fp = tot_false - tot_false_correct
        ap = average_precision_score(gt_labels, pred_labels)
        utils.printout(fout, '%d %d %d %d' % (tp, fn, tn, fp))
        utils.printout(fout, 'precision: %f%%' % (tp * 100 / (tp + fp)))
        utils.printout(fout, 'recall: %f%%' % (tp * 100 / (tp + fn)))
        utils.printout(fout, 'AP: %f%%' % (ap * 100))

