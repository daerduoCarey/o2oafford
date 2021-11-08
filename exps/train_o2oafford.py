import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from subprocess import call
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from data import SAPIENVisionDataset
from datagen import DataGen


def train(conf, train_shape_list, train_data_list, val_data_list, all_train_data_list):
    # create training and validation datasets and data loaders
    data_features = ['scene_pc_cam', 'cam2cambase', 'acting_pc_cambase', \
            'rgb', 'rgb_point', 'rgb_start', 'rgb_final', \
            'gt_applicable_img', 'gt_possible_img', 'gt_applicable_pc', \
            'result', 'cur_dir', 'shape_id']
     
    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.Network(conf.feat_dim)
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # load dataset
    train_dataset = SAPIENVisionDataset(conf.category_types, data_features, \
            env_name=conf.env_name, buffer_max_num=conf.buffer_max_num, img_size=conf.img_size, \
            no_true_false_equal=conf.no_true_false_equal, \
            no_aug_neg_data=conf.no_aug_neg_data)
    
    val_dataset = SAPIENVisionDataset(conf.category_types, data_features, \
            env_name=conf.env_name, buffer_max_num=conf.buffer_max_num, img_size=conf.img_size, \
            no_true_false_equal=conf.no_true_false_equal, \
            no_aug_neg_data=conf.no_aug_neg_data)
    val_dataset.load_data(val_data_list)
    utils.printout(conf.flog, str(val_dataset))
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)

    # create a data generator
    datagen = DataGen(conf.env_name, conf.num_processes_for_datagen, conf.flog)

    # sample succ
    if conf.sample_succ:
        sample_succ_list = []
        sample_succ_dirs = []

    # start training
    start_time = time.time()

    last_train_console_log_step, last_val_console_log_step = None, None

    # if resume
    start_epoch = 0
    if conf.resume:
        # figure out the latest epoch to resume
        for item in os.listdir(os.path.join(conf.exp_dir, 'ckpts')):
            if item.endswith('-train_dataset.pth'):
                start_epoch = max(start_epoch, int(item.split('-')[0]))

        # load states for network, optimizer, lr_scheduler, sample_succ_list
        data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % start_epoch))
        network.load_state_dict(data_to_restore)
        data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % start_epoch))
        network_opt.load_state_dict(data_to_restore)
        data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % start_epoch))
        network_lr_scheduler.load_state_dict(data_to_restore)

        # rmdir and make a new dir for the current sample-succ directory
        old_sample_succ_dir = os.path.join(conf.data_dir, 'epoch-%04d_sample-succ' % (start_epoch - 1))
        utils.force_mkdir(old_sample_succ_dir)

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        ### collect data for the current epoch
        if epoch > start_epoch:
            utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Waiting epoch-{epoch} data ]')
            train_data_list = datagen.join_all()
            utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Gathered epoch-{epoch} data ]')
            cur_data_folders = []
            for item in train_data_list:
                item = '/'.join(item.split('/')[:-1])
                if item not in cur_data_folders:
                    cur_data_folders.append(item)
            for cur_data_folder in cur_data_folders:
                with open(os.path.join(cur_data_folder, 'data_tuple_list.txt'), 'w') as fout:
                    for item in train_data_list:
                        if cur_data_folder == '/'.join(item.split('/')[:-1]):
                            fout.write(item.split('/')[-1]+'\n')

            # load offline-generated sample-random data
            for item in all_train_data_list:
                valid_id_l = conf.num_interaction_data_offline + conf.num_interaction_data * (epoch-1)
                valid_id_r = conf.num_interaction_data_offline + conf.num_interaction_data * epoch
                if valid_id_l <= int(item.split('_')[-2]) < valid_id_r:
                    train_data_list.append(item)

        ### start generating data for the next epoch
        # sample succ
        if conf.sample_succ:
            if conf.resume and epoch == start_epoch:
                sample_succ_list = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-sample_succ_list.pth' % start_epoch))
            else:
                torch.save(sample_succ_list, os.path.join(conf.exp_dir, 'ckpts', '%d-sample_succ_list.pth' % epoch))
            for item in sample_succ_list:
                datagen.add_one_recollect_job(item[0], item[1], item[2], item[3], item[4], item[5], item[6])
            sample_succ_list = []
            sample_succ_dirs = []
            cur_sample_succ_dir = os.path.join(conf.data_dir, 'epoch-%04d_sample-succ' % epoch)
            utils.force_mkdir(cur_sample_succ_dir)

        # start all jobs
        datagen.start_all()
        utils.printout(conf.flog, f'  [ {strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Started generating epoch-{epoch+1} data ]')

        ### load data for the current epoch
        if conf.resume and epoch == start_epoch:
            train_dataset = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % start_epoch))
        elif conf.load_preloaded_train_dataset:
            preloaded_train_dataset_path = os.path.join(conf.log_dir, 'exp-%s-model_3d_critic-None-train_3d_critic'%conf.env_name, 'ckpts', '0-train_dataset.pth')
            train_dataset = torch.load(preloaded_train_dataset_path)
            train_dataset.env_name = conf.env_name
            utils.printout(conf.flog, '[load_preloaded_train_dataset] %s' % preloaded_train_dataset_path)
        else:
            train_dataset.load_data(train_data_list)
        utils.printout(conf.flog, str(train_dataset))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
                num_workers=4, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
        train_num_batch = len(train_dataloader)
        if conf.max_iterations_per_epoch is not None:
            train_num_batch = min(train_num_batch, conf.max_iterations_per_epoch)

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            if (conf.max_iterations_per_epoch is not None) and (train_batch_ind > conf.max_iterations_per_epoch):
                break

            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step
            
            # save checkpoint
            if train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    torch.save(train_dataset, os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            total_loss, end_points = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            # sample succ
            if conf.sample_succ:
                network.eval()

                with torch.no_grad():
                    raise ValueError('SampleSucc: Not Implemented Yet!')

            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                        val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                network.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    __ = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                            step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                            log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
           

def forward(batch, data_features, network, conf, \
        is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
        log_console=False, log_tb=False, tb_writer=None, lr=None):
    # prepare input
    input_scene_pcs_cam = torch.cat(batch[data_features.index('scene_pc_cam')], dim=0).to(conf.device)  # B x 3N x 3
    gt_applicable_pc = torch.cat(batch[data_features.index('gt_applicable_pc')], dim=0).to(conf.device) # B x 3N
    batch_size = input_scene_pcs_cam.shape[0]
    # fps to 10K-points
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(input_scene_pcs_cam, conf.num_point_per_shape).long().reshape(-1)           # BN
    input_scene_pcs_cam = input_scene_pcs_cam[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)    # B x N x 3
    gt_applicable_pc = gt_applicable_pc[input_pcid1, input_pcid2].reshape(batch_size, conf.num_point_per_shape)    # B x N
    
    # convert to cambase (z-up), normalize to zero-center
    cam2cambase_rotmats = torch.cat(batch[data_features.index('cam2cambase')], dim=0).to(conf.device)      # B x 3 x 3
    input_scene_pcs_cambase = torch.matmul(input_scene_pcs_cam, cam2cambase_rotmats.permute(0, 2, 1))
    pc_centers = (input_scene_pcs_cambase.max(dim=1, keepdim=True)[0] + input_scene_pcs_cambase.min(dim=1, keepdim=True)[0]) / 2
    input_scene_pcs_cambase -= pc_centers

    input_acting_pcs_cambase = torch.cat(batch[data_features.index('acting_pc_cambase')], dim=0).to(conf.device)    # B x N' x 3
    input_acting_pcs_cam = torch.matmul(input_acting_pcs_cambase, cam2cambase_rotmats)

    # forward through the network
    pred_result_logits, end_points = network(input_scene_pcs_cambase, input_acting_pcs_cambase)     # B x 2, B x F x N

    # prepare gt
    gt_result = torch.Tensor(batch[data_features.index('result')]).long().to(conf.device)     # B

    # prepare viz
    viz_rgb = torch.cat(batch[data_features.index('rgb')], dim=0).to(conf.device)     # B x 3 x H x W
    gt_applicable_img = torch.cat(batch[data_features.index('gt_applicable_img')], dim=0).to(conf.device) # B x H x W
    gt_possible_img = torch.cat(batch[data_features.index('gt_possible_img')], dim=0).to(conf.device)     # B x H x W
    viz_rgb_point = torch.cat(batch[data_features.index('rgb_point')], dim=0).to(conf.device)     # B x 3 x H x W
    viz_rgb_start = torch.cat(batch[data_features.index('rgb_start')], dim=0).to(conf.device)     # B x 3 x H x W
    viz_rgb_final = torch.cat(batch[data_features.index('rgb_final')], dim=0).to(conf.device)     # B x 3 x H x W
 
    # for each type of loss, compute losses per data
    result_loss_per_data = network.critic.get_ce_loss(pred_result_logits, gt_result)
    
    # for each type of loss, compute avg loss per batch
    result_loss = result_loss_per_data.mean()

    # compute total loss
    total_loss = result_loss

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{data_split:^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{total_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('total_loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)

        # inference all results
        #pred_all_result = network.inference_whole_pc(input_scene_pcs_cambase, input_acting_pcs_cambase)
        pred_all_result = torch.zeros_like(input_scene_pcs_cambase)[:, :, 0]
        # remove all non-applicable pixels from viz
        pred_all_result[~gt_applicable_pc] = 0

        # gen visu
        if is_val and (not conf.no_visu) and epoch % conf.num_epoch_every_visu == 0:
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'epoch-%04d' % epoch)
            input_scene_pc_cam_dir = os.path.join(out_dir, 'input_scene_pc_cam')
            input_scene_pc_cambase_dir = os.path.join(out_dir, 'input_scene_pc_cambase')
            input_acting_pc_cam_dir = os.path.join(out_dir, 'input_acting_pc_cam')
            input_acting_pc_cambase_dir = os.path.join(out_dir, 'input_acting_pc_cambase')
            pred_all_result_dir = os.path.join(out_dir, 'pred_all_result')
            viz_rgb_dir = os.path.join(out_dir, 'viz_rgb')
            gt_applicable_dir = os.path.join(out_dir, 'gt_applicable')
            gt_possible_dir = os.path.join(out_dir, 'gt_possible')
            viz_rgb_point_dir = os.path.join(out_dir, 'viz_rgb_point')
            viz_rgb_start_dir = os.path.join(out_dir, 'viz_rgb_start')
            viz_rgb_final_dir = os.path.join(out_dir, 'viz_rgb_final')
            info_dir = os.path.join(out_dir, 'info')
            
            if batch_ind == 0:
                # create folders
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                os.mkdir(out_dir)
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

            if batch_ind < conf.num_batch_every_visu:
                utils.printout(conf.flog, 'Visualizing ...')
                for i in range(batch_size):
                    fn = 'data-%03d.png' % (batch_ind * batch_size + i)
                    utils.render_pts(os.path.join(BASE_DIR, input_scene_pc_cam_dir, fn.split('.')[0]), input_scene_pcs_cam[i].cpu().numpy(), highlight_id=0, campos=0)
                    utils.render_pts(os.path.join(BASE_DIR, input_scene_pc_cambase_dir, fn.split('.')[0]), input_scene_pcs_cambase[i].cpu().numpy(), highlight_id=0)
                    utils.render_pts(os.path.join(BASE_DIR, input_acting_pc_cam_dir, fn.split('.')[0]), input_acting_pcs_cam[i].cpu().numpy())
                    utils.render_pts(os.path.join(BASE_DIR, input_acting_pc_cambase_dir, fn.split('.')[0]), input_acting_pcs_cambase[i].cpu().numpy())
                    utils.render_pts_label_png(os.path.join(BASE_DIR, pred_all_result_dir, fn.split('.')[0]), input_scene_pcs_cam[i].cpu().numpy(), pred_all_result[i].cpu().numpy(), campos=0)
                    img_toshow = (viz_rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_toshow).save(os.path.join(viz_rgb_dir, fn))
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
                        fout.write('cur_dir: %s\n' % batch[data_features.index('cur_dir')][i])
                        fout.write('pred: %s\n' % utils.print_true_false((pred_result_logits[i]>0).cpu().numpy()))
                        fout.write('gt: %s\n' % utils.print_true_false(gt_result[i].cpu().numpy()))
                        fout.write('result_loss: %f\n' % result_loss_per_data[i].item())
                
            if batch_ind == conf.num_batch_every_visu - 1:
                # visu html
                utils.printout(conf.flog, 'Generating html visualization ...')
                sublist = 'input_scene_pc_cambase,input_acting_pc_cambase,input_scene_pc_cam,input_acting_pc_cam,pred_all_result,viz_rgb,gt_applicable,gt_possible,viz_rgb_point,viz_rgb_start,viz_rgb_final,info'
                cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (out_dir, os.path.join(BASE_DIR, 'gen_html_hierachy_local.py'), sublist, sublist)
                call(cmd, shell=True)
                utils.printout(conf.flog, 'DONE')

    for k in end_points:
        end_points[k] = end_points[k].detach()

    return total_loss, end_points


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--env_name', type=str, help='env name')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all categories]', default=None)
    parser.add_argument('--data_dir_prefix', type=str, help='data directory', default=None)
    parser.add_argument('--offline_data_dir', type=str, help='data directory', default=None)
    parser.add_argument('--val_data_dir', type=str, help='data directory', default=None)
    parser.add_argument('--val_data_fn', type=str, help='data directory', default='data_tuple_list.txt')
    parser.add_argument('--val_max_num_data', type=int, help='max num data for val', default=1000)

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    #parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--buffer_max_num', type=int, default=None)
    parser.add_argument('--max_iterations_per_epoch', type=int, default=5000)
    parser.add_argument('--num_processes_for_datagen', type=int, default=20)
    parser.add_argument('--num_interaction_data_offline', type=int, default=5)
    parser.add_argument('--num_interaction_data', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--sample_succ', action='store_true', default=False)
    parser.add_argument('--load_preloaded_train_dataset', action='store_true', default=False)

    # loss weights

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()


    ### prepare before training
    if conf.data_dir_prefix is None:
        conf.data_dir_prefix = '../data/offlinedata-%s' % conf.env_name
    if conf.offline_data_dir is None:
        conf.offline_data_dir = conf.data_dir_prefix + '-train_cat_train_shape'
    if conf.val_data_dir is None:
        conf.val_data_dir = conf.data_dir_prefix + '-train_cat_test_shape'

    # make exp_name
    conf.exp_name = f'exp-{conf.env_name}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.mkdir(conf.exp_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        if not conf.no_visu:
            os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # prepare data_dir
    conf.data_dir = os.path.join('../data/', conf.exp_name)
    if os.path.exists(conf.data_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A data_dir named "%s" already exists, overwrite? (y/n) ' % conf.data_dir)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.data_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no data_dir named %s to resume!' % conf.data_dir)
    if not conf.resume:
        os.mkdir(conf.data_dir)

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # no_aug_neg_data
    conf.no_aug_neg_data = False
    if conf.env_name in ['rotating', 'pushing', 'stacking']:
        conf.no_aug_neg_data = True

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')
    utils.printout(flog, f'offline_data_dir: {conf.offline_data_dir}')
    utils.printout(flog, f'val_data_dir: {conf.val_data_dir}')

    # backup python files used for this training
    if not conf.resume:
        os.system('cp datagen.py data.py models/%s.py %s %s' % (conf.model_version, __file__, conf.exp_dir))
     
    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device
    
    # parse params
    if conf.category_types is not None:
        conf.category_types = conf.category_types.split(',')
    utils.printout(flog, 'category_types: %s' % str(conf.category_types))
    
    # read cat2freq
    conf.cat2freq = dict()
    with open('../stats/all_cats_cnt_freq.txt', 'r') as fin:
        for l in fin.readlines():
            category, _, freq = l.rstrip().split()
            conf.cat2freq[category] = int(freq)
    utils.printout(flog, str(conf.cat2freq))

    # load train cats
    with open(os.path.join('env_%s' % conf.env_name, 'stats', 'afford_cats-train.txt'), 'r') as fin:
        cats = [l.rstrip() for l in fin.readlines()]

    # read train_shape_fn
    train_shape_list = []
    for cat in cats:
        with open('../stats/%s-train_cat_train_shape.txt' % cat, 'r') as fin:
            for l in fin.readlines():
                train_shape_list.append((l.rstrip(), cat))
    utils.printout(flog, 'len(train_shape_list): %d' % len(train_shape_list))
    
    with open(os.path.join(conf.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(conf.offline_data_dir, l.rstrip()) for l in fin.readlines()]
    utils.printout(flog, 'len(all_train_data_list): %d' % len(all_train_data_list))
    if conf.resume:
        train_data_list = None
    else:
        train_data_list = []
        for item in all_train_data_list:
            if int(item.split('_')[-2]) < conf.num_interaction_data_offline:
                train_data_list.append(item)
        utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))
    
    with open(os.path.join(conf.val_data_dir, conf.val_data_fn), 'r') as fin:
        val_data_list = [os.path.join(conf.val_data_dir, l.rstrip()) for l in fin.readlines()]
    val_data_list = val_data_list[:conf.val_max_num_data]
    utils.printout(flog, 'len(val_data_list): %d' % len(val_data_list))
     
    ### start training
    train(conf, train_shape_list, train_data_list, val_data_list, all_train_data_list)


    ### before quit
    # close file log
    flog.close()

