import os
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import math

from hybrik.utils.config import update_config

import my_config
import subsets

from hybrik.utils.config import update_config
from hybrik.models import builder
from hybrik.utils.pose_utils import scale_and_translation_transform_batch, check_joints2d_visibility_torch, compute_similarity_transform_batch
from hybrik.utils.renderer_mine import Renderer
from hybrik.utils.geometry import convert_weak_perspective_to_camera_translation, undo_keypoint_normalisation, orthographic_project_torch
from hybrik.models.smpl_mine import SMPL
from hybrik.datasets.pw3d_eval_dataset import PW3DEvalDataset


def evaluate_3dpw(model,
                  model_cfg,
                  eval_dataset,
                  metrics_to_track,
                  device,
                  save_path,
                  num_workers=4,
                  pin_memory=True,
                  vis_img_wh=512,
                  vis_every_n_batches=1000,
                  extreme_crop=False):
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl_neutral = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1).to(device)
    smpl_male = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1, gender='male').to(device)
    smpl_female = SMPL(my_config.SMPL_MODEL_DIR, batch_size=1, gender='female').to(device)

    metric_sums = {'num_datapoints': 0}
    per_frame_metrics = {}
    for metric in metrics_to_track:
        metric_sums[metric] = 0.
        per_frame_metrics[metric] = []

        if metric == 'hrnet_joints2D_l2es':
            metric_sums['num_vis_hrnet_joints2D'] = 0

        elif metric == 'joints2D_l2es':
            metric_sums['num_vis_joints2D'] = 0


    fname_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []


    renderer = Renderer(img_res=vis_img_wh, faces=smpl_neutral.faces)
    reposed_cam_t = convert_weak_perspective_to_camera_translation(cam_wp=np.array([0.85, 0., -0.2]),
                                                                   focal_length=5000.,
                                                                   resolution=vis_img_wh)
    if extreme_crop:
        rot_cam_t = convert_weak_perspective_to_camera_translation(cam_wp=np.array([0.85, 0., 0.]),
                                                                   focal_length=5000.,
                                                                   resolution=vis_img_wh)

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # if batch_num == 2:
        #     break
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input'].to(device)
        target_pose = samples_batch['pose'].to(device)
        target_shape = samples_batch['shape'].to(device)
        target_gender = samples_batch['gender'][0]
        hrnet_joints2D_coco = samples_batch['hrnet_kps']
        hrnet_joints2D_vis_coco = samples_batch['hrnet_kps_vis']
        hrnet_joints2D_vis_coco = check_joints2d_visibility_torch(hrnet_joints2D_coco,
                                                                  input.shape[-1],
                                                                  vis=hrnet_joints2D_vis_coco)  # (batch_size, 17)
        target_joints2D_coco = samples_batch['gt_kps']
        target_joints2D_vis_coco = samples_batch['gt_kps_vis']
        target_joints2D_vis_coco = check_joints2d_visibility_torch(target_joints2D_coco,
                                                                   input.shape[-1],
                                                                   vis=target_joints2D_vis_coco)  # (batch_size, 17)
        fname = samples_batch['fname']

        if target_gender == 'm':
            target_smpl_output = smpl_male(body_pose=target_pose[:, 3:],
                                           global_orient=target_pose[:, :3],
                                           betas=target_shape)
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_smpl_output = smpl_female(body_pose=target_pose[:, 3:],
                                             global_orient=target_pose[:, :3],
                                             betas=target_shape)
            target_reposed_smpl_output = smpl_female(betas=target_shape)

        target_vertices = target_smpl_output.vertices
        target_joints_h36mlsp = target_smpl_output.joints[:, my_config.ALL_JOINTS_TO_H36M_MAP, :][:, my_config.H36M_TO_J14, :]
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------- PREDICTIONS -------------------------------
        out = model(input)
        # for key in out:
        #     print(key, out[key].shape)
        """
        out contains
        pred_phi torch.Size([1, 23, 2])
        pred_delta_shape torch.Size([1, 10])
        pred_shape torch.Size([1, 10])
        pred_theta_mats torch.Size([1, 24, 3, 3])
        pred_uvd_jts torch.Size([1, 87])
        pred_xyz_jts_29 torch.Size([1, 87])
        pred_xyz_jts_24 torch.Size([1, 72])
        pred_xyz_jts_24_struct torch.Size([1, 72])
        pred_xyz_jts_17 torch.Size([1, 51])
        pred_vertices torch.Size([1, 6890, 3])
        maxvals torch.Size([1, 29, 1])
        cam_scale torch.Size([1, 1])
        cam_trans torch.Size([1, 2])
        cam_root torch.Size([1, 3])
        transl torch.Size([1, 3])
        pred_camera torch.Size([1, 1])
        sigma torch.Size([1, 29, 1])
        scores torch.Size([1, 29, 1])
        img_feat torch.Size([1, 2048])

        """
        pred_cam_wp = torch.cat([out.cam_scale * 2, out.cam_trans], axis=-1)  # TODO camera doesn't work like that, need to use their own code probably
        pred_pose_rotmats = out.pred_theta_mats
        pred_shape = out.pred_shape

        pred_smpl_output = smpl_neutral(body_pose=pred_pose_rotmats[:, 1:, :, :],
                                        global_orient=pred_pose_rotmats[:, [0], :, :],
                                        betas=pred_shape,
                                        pose2rot=False)
        pred_vertices = pred_smpl_output.vertices  # (1, 6890, 3)
        pred_joints_h36mlsp = pred_smpl_output.joints[:, my_config.ALL_JOINTS_TO_H36M_MAP, :][:, my_config.H36M_TO_J14, :]  # (1, 14, 3)
        pred_joints_coco = pred_smpl_output.joints[:, my_config.ALL_JOINTS_TO_COCO_MAP, :]  # (1, 17, 3)

        pred_vertices2D_for_vis = orthographic_project_torch(pred_vertices, pred_cam_wp, scale_first=False)
        pred_vertices2D_for_vis = undo_keypoint_normalisation(pred_vertices2D_for_vis, vis_img_wh)
        pred_joints2D_coco_normed = orthographic_project_torch(pred_joints_coco, pred_cam_wp)  # (1, 17, 2)
        pred_joints2D_coco = undo_keypoint_normalisation(pred_joints2D_coco_normed, input.shape[-1])
        pred_joints2D_coco_for_vis = undo_keypoint_normalisation(pred_joints2D_coco_normed, vis_img_wh)

        pred_reposed_vertices = smpl_neutral(betas=pred_shape).vertices  # (1, 6890, 3)

        # ------------------------------------------------ METRICS ------------------------------------------------

        # Numpy-fying targets
        target_vertices = target_vertices.cpu().detach().numpy()
        target_joints_h36mlsp = target_joints_h36mlsp.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
        hrnet_joints2D_coco = hrnet_joints2D_coco.cpu().detach().numpy()
        hrnet_joints2D_vis_coco = hrnet_joints2D_vis_coco.cpu().detach().numpy()
        target_joints2D_coco = target_joints2D_coco.cpu().detach().numpy()
        target_joints2D_vis_coco = target_joints2D_vis_coco.cpu().detach().numpy()

        # Numpy-fying preds
        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_joints_h36mlsp = pred_joints_h36mlsp.cpu().detach().numpy()
        pred_joints_coco = pred_joints_coco.cpu().detach().numpy()
        pred_vertices2D_for_vis = pred_vertices2D_for_vis.cpu().detach().numpy()
        pred_joints2D_coco = pred_joints2D_coco.cpu().detach().numpy()
        pred_joints2D_coco_for_vis = pred_joints2D_coco_for_vis.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()

        # -------------- 3D Metrics with Mode and Minimum Error Samples --------------
        if 'pves' in metrics_to_track:
            pve_batch = np.linalg.norm(pred_vertices - target_vertices,
                                       axis=-1)  # (bs, 6890)
            metric_sums['pves'] += np.sum(pve_batch)  # scalar
            per_frame_metrics['pves'].append(np.mean(pve_batch, axis=-1))

        # Scale and translation correction
        if 'pves_sc' in metrics_to_track:
            pred_vertices_sc = scale_and_translation_transform_batch(
                pred_vertices,
                target_vertices)
            pve_sc_batch = np.linalg.norm(
                pred_vertices_sc - target_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pves_sc'] += np.sum(pve_sc_batch)  # scalar
            per_frame_metrics['pves_sc'].append(np.mean(pve_sc_batch, axis=-1))

        # Procrustes analysis
        if 'pves_pa' in metrics_to_track:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bs, 6890)
            metric_sums['pves_pa'] += np.sum(pve_pa_batch)  # scalar
            per_frame_metrics['pves_pa'].append(np.mean(pve_pa_batch, axis=-1))

        if 'pve-ts' in metrics_to_track:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            metric_sums['pve-ts'] += np.sum(pvet_batch)  # scalar
            per_frame_metrics['pve-ts'].append(np.mean(pvet_batch, axis=-1))

        # Scale and translation correction
        if 'pve-ts_sc' in metrics_to_track:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices,
                target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(
                pred_reposed_vertices_sc - target_reposed_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pve-ts_sc'] += np.sum(pvet_scale_corrected_batch)  # scalar
            per_frame_metrics['pve-ts_sc'].append(np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'mpjpes' in metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_joints_h36mlsp - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            metric_sums['mpjpes'] += np.sum(mpjpe_batch)  # scalar
            per_frame_metrics['mpjpes'].append(np.mean(mpjpe_batch, axis=-1))

        # Scale and translation correction
        if 'mpjpes_sc' in metrics_to_track:
            pred_joints_h36mlsp_sc = scale_and_translation_transform_batch(
                pred_joints_h36mlsp,
                target_joints_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(
                pred_joints_h36mlsp_sc - target_joints_h36mlsp,
                axis=-1)  # (bs, 14)
            metric_sums['mpjpes_sc'] += np.sum(mpjpe_sc_batch)  # scalar
            per_frame_metrics['mpjpes_sc'].append(np.mean(mpjpe_sc_batch, axis=-1))

        # Procrustes analysis
        if 'mpjpes_pa' in metrics_to_track:
            pred_joints_h36mlsp_pa = compute_similarity_transform_batch(pred_joints_h36mlsp, target_joints_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints_h36mlsp_pa - target_joints_h36mlsp, axis=-1)  # (bs, 14)
            metric_sums['mpjpes_pa'] += np.sum(mpjpe_pa_batch)  # scalar
            per_frame_metrics['mpjpes_pa'].append(np.mean(mpjpe_pa_batch, axis=-1))

        # -------------------------------- 2D Metrics ---------------------------
        # Using HRNet 2D joints as target, rather than GT
        if 'hrnet_joints2D_l2es' in metrics_to_track:
            hrnet_joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco[:, hrnet_joints2D_vis_coco[0], :] - hrnet_joints2D_coco[:, hrnet_joints2D_vis_coco[0], :],
                                                      axis=-1)  # (1, num vis joints)
            assert hrnet_joints2D_l2e_batch.shape[1] == hrnet_joints2D_vis_coco.sum()

            metric_sums['hrnet_joints2D_l2es'] += np.sum(hrnet_joints2D_l2e_batch)  # scalar
            metric_sums['num_vis_hrnet_joints2D'] += hrnet_joints2D_l2e_batch.shape[1]
            per_frame_metrics['hrnet_joints2D_l2es'].append(np.mean(hrnet_joints2D_l2e_batch, axis=-1))  # (1,)

        if 'joints2D_l2es' in metrics_to_track:
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco[:, target_joints2D_vis_coco[0], :] - target_joints2D_coco[:, target_joints2D_vis_coco[0], :],
                                                axis=-1)  # (1, num vis joints)
            assert joints2D_l2e_batch.shape[1] == target_joints2D_vis_coco.sum()

            metric_sums['joints2D_l2es'] += np.sum(joints2D_l2e_batch)  # scalar
            metric_sums['num_vis_joints2D'] += joints2D_l2e_batch.shape[1]
            per_frame_metrics['joints2D_l2es'].append(np.mean(joints2D_l2e_batch, axis=-1))  # (1,)

        metric_sums['num_datapoints'] += target_pose.shape[0]

        fname_per_frame.append(fname)
        pose_per_frame.append(pred_pose_rotmats.cpu().detach().numpy())
        shape_per_frame.append(pred_shape.cpu().detach().numpy())
        cam_per_frame.append(pred_cam_wp.cpu().detach().numpy())

        # ------------------------------- VISUALISE -------------------------------
        if vis_every_n_batches is not None and batch_num % vis_every_n_batches == 0:
            vis_img = samples_batch['vis_img'].numpy()

            # pred_cam_t = out['pred_cam_t'][0, 0, :].cpu().detach().numpy()
            pred_cam_t = torch.stack([pred_cam_wp[0, 1],
                                      pred_cam_wp[0, 2],
                                      2 * 5000 / (vis_img_wh * pred_cam_wp[0, 0] + 1e-9)], dim=-1).cpu().detach().numpy()

            # Render predicted meshes
            body_vis_rgb_mode = renderer(vertices=pred_vertices[0],
                                         camera_translation=pred_cam_t.copy(),
                                         image=vis_img[0])
            body_vis_rgb_mode_rot = renderer(vertices=pred_vertices[0],
                                             camera_translation=pred_cam_t.copy() if not extreme_crop else rot_cam_t.copy(),
                                             image=np.zeros_like(vis_img[0]),
                                             angle=np.pi / 2.,
                                             axis=[0., 1., 0.])

            reposed_body_vis_rgb_mean = renderer(vertices=pred_reposed_vertices[0],
                                                 camera_translation=reposed_cam_t.copy(),
                                                 image=np.zeros_like(vis_img[0]),
                                                 flip_updown=False)
            reposed_body_vis_rgb_mean_rot = renderer(vertices=pred_reposed_vertices[0],
                                                     camera_translation=reposed_cam_t.copy(),
                                                     image=np.zeros_like(vis_img[0]),
                                                     angle=np.pi / 2.,
                                                     axis=[0., 1., 0.],
                                                     flip_updown=False)

            # ------------------ Model Prediction, Error and Uncertainty Figure ------------------
            num_row = 6
            num_col = 6
            subplot_count = 1
            plt.figure(figsize=(20, 20))

            # Plot image and mask vis
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(vis_img[0])
            subplot_count += 1

            # Plot pred vertices 2D and body render overlaid over input
            # also add target joints 2D scatter
            target_joints2D_coco = target_joints2D_coco * (vis_img_wh / input.shape[-1])
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(vis_img[0])
            plt.scatter(pred_vertices2D_for_vis[0, :, 0],
                        pred_vertices2D_for_vis[0, :, 1],
                        c='r', s=0.01)
            if 'joints2D_l2es' in metrics_to_track:
                plt.scatter(pred_joints2D_coco_for_vis[0, :, 0],
                            pred_joints2D_coco_for_vis[0, :, 1],
                            c='r', s=10.0)
                for j in range(target_joints2D_coco.shape[1]):
                    if target_joints2D_vis_coco[0][j]:
                        plt.scatter(target_joints2D_coco[0, j, 0],
                                    target_joints2D_coco[0, j, 1],
                                    c='blue', s=10.0)
                        plt.text(target_joints2D_coco[0, j, 0],
                                 target_joints2D_coco[0, j, 1],
                                 str(j))
                    plt.text(pred_joints2D_coco_for_vis[0, j, 0],
                             pred_joints2D_coco_for_vis[0, j, 1],
                             str(j))
            subplot_count += 1

            # Plot body render overlaid on vis image
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_mode)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_mode_rot)
            subplot_count += 1

            # Plot reposed body render
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(reposed_body_vis_rgb_mean)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(reposed_body_vis_rgb_mean_rot)
            subplot_count += 1

            if 'pves_sc' in metrics_to_track:
                # Plot PVE-SC pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][batch_num][0]))
                subplot_count += 1

            if 'pves_pa' in metrics_to_track:
                # Plot PVE-PA pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-PA')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][batch_num][0]))
                subplot_count += 1

            if 'pve-ts_sc' in metrics_to_track:
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-T-SC')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_reposed_vertices[0, :, 0],
                            target_reposed_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_reposed_vertices_sc[0, :, 0],
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(pred_reposed_vertices_sc[0, :, 0],
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pvet_scale_corrected_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(pred_reposed_vertices_sc[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pvet_scale_corrected_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][batch_num][0]))
                subplot_count += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            save_fig_path = os.path.join(save_path, fname[0])
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()


    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    print('\n--- Check Pred save Shapes ---')
    fname_per_frame = np.concatenate(fname_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)
    print(fname_per_frame.shape)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)
    print(pose_per_frame.shape)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)
    print(shape_per_frame.shape)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)
    print(cam_per_frame.shape)

    final_metrics = {}
    for metric_type in metrics_to_track:

        if metric_type == 'hrnet_joints2D_l2es':
            joints2D_l2e = metric_sums['hrnet_joints2D_l2es'] / metric_sums['num_vis_hrnet_joints2D']
            final_metrics[metric_type] = joints2D_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_hrnet_joints2D'])

        elif metric_type == 'joints2D_l2es':
            joints2D_l2e = metric_sums['joints2D_l2es'] / metric_sums['num_vis_joints2D']
            final_metrics[metric_type] = joints2D_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_joints2D'])

        else:
            if 'pves' in metric_type:
                num_per_sample = 6890
            elif 'mpjpes' in metric_type:
                num_per_sample = 14
            # print('Check total samples:', metric_type, num_per_sample, self.total_samples)
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_per_sample)

    print('\n---- Metrics ----')
    for metric in final_metrics.keys():
        if final_metrics[metric] > 0.3:
            mult = 1
        else:
            mult = 1000
        print(metric, '{:.2f}'.format(final_metrics[metric] * mult))  # Converting from metres to millimetres

    print('\n---- Check metric save shapes ----')
    for metric_type in metrics_to_track:
        per_frame = np.concatenate(per_frame_metrics[metric_type], axis=0)
        print(metric_type, per_frame.shape)
        np.save(os.path.join(save_path, metric_type + '_per_frame.npy'), per_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./pretrained_hrnet.pth', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_cfg', type=str, default='configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml')
    parser.add_argument('--gpu', default='0', type=str, help='GPU')
    parser.add_argument('--use_subset', '-S', action='store_true')
    parser.add_argument('--extreme_crop', '-C', action='store_true')
    parser.add_argument('--extreme_crop_scale', '-CS', type=float, default=0.5)

    args = parser.parse_args()

    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model
    model_cfg = update_config(args.model_cfg)
    hybrik_model = builder.build_sppe(model_cfg.MODEL).to(device)
    hybrik_model.eval()
    print(f'Loading model from {args.checkpoint}...')
    save_dict = torch.load(args.checkpoint, map_location=device)
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)

    # Setup evaluation dataset
    if args.use_subset:
        selected_fnames = subsets.PW3D_OCCLUDED_JOINTS
        vis_every_n_batches = 1
        vis_joints_threshold = 0.8
    else:
        selected_fnames = None
        vis_every_n_batches = 1000
        vis_joints_threshold = 0.6
    # vis_every_n_batches=None
    dataset_path = '/scratches/nazgul_2/as2562/datasets/3DPW/test'
    dataset = PW3DEvalDataset(dataset_path,
                              img_wh=model_cfg.MODEL.IMAGE_SIZE[0],
                              selected_fnames=selected_fnames,
                              visible_joints_threshold=vis_joints_threshold,
                              gt_visible_joints_threhshold=0.6,
                              extreme_crop=args.extreme_crop,
                              extreme_crop_scale=args.extreme_crop_scale,
                              vis_img_wh=512)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc', 'mpjpes', 'mpjpes_sc', 'mpjpes_pa']
    metrics.append('hrnet_joints2D_l2es')
    metrics.append('joints2D_l2es')

    save_path = '/scratch3/as2562/HybrIK/evaluations/3dpw'
    if args.use_subset:
        save_path += '_selected_fnames_occluded_joints'
    if args.extreme_crop:
        save_path += '_extreme_crop_scale_{}'.format(args.extreme_crop_scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Saving to:", save_path)

    # Run evaluation
    evaluate_3dpw(model=hybrik_model,
                  model_cfg=model_cfg,
                  eval_dataset=dataset,
                  metrics_to_track=metrics,
                  device=device,
                  save_path=save_path,
                  num_workers=4,
                  pin_memory=True,
                  vis_every_n_batches=vis_every_n_batches,
                  vis_img_wh=512,
                  extreme_crop=args.extreme_crop)

