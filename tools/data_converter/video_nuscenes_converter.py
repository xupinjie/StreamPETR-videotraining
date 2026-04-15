# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset
import tqdm
import cv2
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import pickle


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


def _match_scenes_by_split(available_scenes, split_names):
    """Return a set of scene tokens whose name (or base name without _x{n} suffix)
    appears in split_names.  Handles datasets duplicated by create_nusc_mini_x8_dataset.py
    where scene names have an appended _x{n} suffix.
    """
    import re
    split_set = set(split_names)
    tokens = set()
    for scene in available_scenes:
        name = scene['name']
        base = re.sub(r'_x\d+$', '', name)
        if name in split_set or base in split_set:
            tokens.add(scene['token'])
    return tokens


def create_nuscenes_infos(root_path,
                          out_dir,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          encode_mjpeg=True):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    train_scenes = _match_scenes_by_split(available_scenes, train_scenes)
    val_scenes   = _match_scenes_by_split(available_scenes, val_scenes)

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    valid_channels = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    all_scenes = sorted(list((train_scenes | val_scenes)))
    print("==>All scenes to process: ", [nusc.get('scene', scene_token)["name"] for scene_token in all_scenes])
    if encode_mjpeg:
        os.makedirs(os.path.join(root_path, f"{out_dir}/video_samples_mjpeg"), exist_ok=True)
    if not os.path.exists('nusc.pkl'):
        for scene in tqdm.tqdm(all_scenes):
            for channel in valid_channels:
                if encode_mjpeg:
                    out_path = f"{out_dir}/video_samples_mjpeg/{scene}__{channel}.mp4"
                else:
                    out_path = osp.join(root_path, 'video_samples', f'{scene}__{channel}.mp4')
                render_scene_channel_video(nusc, scene,
                                           channel, imsize=(1600, 900),
                                           out_path=out_path)

        with open('nusc.pkl', 'wb') as file:
            pickle.dump(nusc, file)
    else:
        print("Loading nusc from file!!!!")
        with open('nusc.pkl', 'rb') as file:
            nusc = pickle.load(file)
    print(f"Is test: {test} | version = {version}")
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    print("=============_fill_trainval_infos done!")

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, out_dir,
                             '{}_temporal_infos_test.pkl'.format(info_prefix))
        if not os.path.exists(info_path):
            mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, out_dir,
                             '{}_temporal_infos_train.pkl'.format(info_prefix))
        if not os.path.exists(info_path):
            mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path, out_dir,
                                 '{}_temporal_infos_val.pkl'.format(info_prefix))
        if not os.path.exists(info_val_path):
            mmcv.dump(data, info_val_path)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            root_path, out_dir, f'{info_prefix}_temporal_infos_test.pkl')
        export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(
            root_path, out_dir, f'{info_prefix}_temporal_infos_train.pkl')
        info_val_path = osp.join(
            root_path, out_dir, f'{info_prefix}_temporal_infos_val.pkl')
        export_2d_annotation(
            root_path, info_train_path, version=version)
        export_2d_annotation(
            root_path, info_val_path, version=version)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class."""
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data."""
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'sweeps': [],
            'frame_idx': frame_idx,
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            video_path, video_frame_id = get_sample_video_data(nusc, cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            cam_info.update(video_path=video_path, video_frame_id=video_frame_id)
            cam_info.update(cam_path=cam_path)
            info['cams'].update({cam: cam_info})

        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

            gt_2dbboxes_cams = []
            gt_3dbboxes_cams = []
            centers2d_cams = []
            gt_2dbboxes_ignore_cams = []
            gt_2dlabels_cams = []
            depths_cams = []
            visibilities = []

            for cam_type, cam_info in info['cams'].items():
                gt_3dbboxes = []
                gt_2dbboxes = []
                centers2d = []
                gt_2dbboxes_ignore = []
                gt_2dlabels = []
                depths = []
                visibility = []
                try:
                    (height, width, _) = mmcv.imread(cam_info['data_path']).shape
                except:
                    print(f"imread: {cam_info['data_path']}")
                    continue
                annos_cam = get_2d_boxes(nusc, cam_info['sample_data_token'], visibilities=['', '1', '2', '3', '4'], mono3d=True)
                for i, ann in enumerate(annos_cam):
                    if ann is None:
                        continue
                    if ann.get('ignore', False):
                        continue
                    x1, y1, w, h = ann['bbox']
                    inter_w = max(0, min(x1 + w, width) - max(x1, 0))
                    inter_h = max(0, min(y1 + h, height) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if ann['area'] <= 0 or w < 1 or h < 1:
                        continue
                    if ann['category_name'] not in nus_categories:
                        continue
                    bbox = [x1, y1, x1 + w, y1 + h]
                    if ann.get('iscrowd', False):
                        gt_2dbboxes_ignore.append(bbox)
                    else:
                        gt_2dbboxes.append(bbox)
                        gt_2dlabels.append(ann['category_id'])
                        center2d = ann['center2d'][:2]
                        depth = ann['center2d'][2]
                        centers2d.append(center2d)
                        depths.append(depth)
                        visibility.append(ann['visibility_token'])
                        gt_3dbboxes.append(ann['bbox_cam3d'])
                gt_2dbboxes = np.array(gt_2dbboxes, dtype=np.float32)
                gt_3dbboxes_cam = np.array(gt_3dbboxes, dtype=np.float32)
                gt_2dlabels = np.array(gt_2dlabels, dtype=np.int64)
                centers2d = np.array(centers2d, dtype=np.float32)
                depths = np.array(depths, dtype=np.float32)
                gt_2dbboxes_ignore = np.array(gt_2dbboxes_ignore, dtype=np.float32)
                gt_2dbboxes_cams.append(gt_2dbboxes)
                gt_2dlabels_cams.append(gt_2dlabels)
                centers2d_cams.append(centers2d)
                gt_3dbboxes_cams.append(gt_3dbboxes_cam)
                depths_cams.append(depths)
                gt_2dbboxes_ignore_cams.append(gt_2dbboxes_ignore)
                visibilities.append(visibility)
                info.update(
                    dict(
                        bboxes2d=gt_2dbboxes_cams,
                        bboxes3d_cams=gt_3dbboxes_cams,
                        labels2d=gt_2dlabels_cams,
                        centers2d=centers2d_cams,
                        depths=depths_cams,
                        bboxes_ignore=gt_2dbboxes_ignore_cams,
                        visibilities=visibilities,)
                )

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matrix from general sensor to Top LiDAR."""
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data."""
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            try:
                coco_infos = get_2d_boxes(
                    nusc,
                    cam_info['sample_data_token'],
                    visibilities=['', '1', '2', '3', '4'],
                    mono3d=mono3d)
            except:
                print(f"error, continue to next")
                continue
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')[-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`."""
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        box = nusc.get_box(ann_rec['token'])

        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        final_coords = post_process_coords(corner_coords)

        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection."""
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record."""
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0
    coco_rec['visibility_token'] = repro_rec['visibility_token']

    return coco_rec


def render_scene_channel_video(nusc, scene_token: str,
                               channel: str = 'CAM_FRONT',
                               freq: float = 10,
                               imsize: Tuple[float, float] = (640, 360),
                               out_path: str = None) -> None:
    valid_channels = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    assert channel in valid_channels, 'Error: Input channel {} not valid.'.format(channel)
    scene_rec = nusc.get('scene', scene_token)
    sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', sample_rec['data'][channel])
    video_abs_path = osp.join(nusc.dataroot, out_path)

    video_regenerate = True
    if osp.exists(video_abs_path):
        cap = cv2.VideoCapture(video_abs_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames > 0:
            video_regenerate = False
            print(f"=>video_out_path={video_abs_path} works fine, frames={frames}")
        else:
            print(f"{video_abs_path} exists, but is broken. Need to regenerate.")
            os.remove(video_abs_path)
    out = None
    if out_path is not None and video_regenerate:
        print(f"Generating new video file: {video_abs_path}")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_abs_path, fourcc, freq, imsize)
    has_more_frames = True
    video_frame_id = 0

    image_cnt = 0
    while has_more_frames:
        impath, boxes, camera_intrinsic = nusc.get_sample_data(sd_rec['token'],
                                                               box_vis_level=BoxVisibility.ANY)
        image_cnt += 1
        if not osp.exists(impath):
            raise Exception('Error: Missing image %s' % impath)

        if out is not None:
            im = cv2.imread(impath)
            out.write(im)
        sd_rec.update(
            video_path=out_path,
            video_frame_id=video_frame_id
        )
        video_frame_id = video_frame_id + 1
        if not sd_rec['next'] == "":
            sd_rec = nusc.get('sample_data', sd_rec['next'])
        else:
            has_more_frames = False
    if out is not None:
        out.release()


def get_sample_video_data(nusc, sample_data_token):
    sd_record = nusc.get('sample_data', sample_data_token)
    if 'video_path' in sd_record:
        return osp.join(nusc.dataroot, sd_record['video_path']), sd_record['video_frame_id']
    else:
        return None, None
