# In order to import model
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

#from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
#from mmpose.datasets import build_dataloader, build_dataset
from models import build_posenet

import numpy as np
from torchvision.transforms import functional as F
from mmpose.core.post_processing import get_affine_transform
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def add_joints(image, joints):
    coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r']

    coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)}

    coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')]

    part_idx = coco_part_idx
    part_orders = coco_part_orders

    def link(a, b):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
 
            if jointa[2] > 0.2 and jointb[2] > 0.2:
                cv2.line(image, (int(jointa[0]), int(jointa[1])), (int(jointb[0]), int(jointb[1])), (0, 0, 255), 2)

    # add joints
    for joint in joints:
        if joint[2] > 0.2:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 2, (0, 255, 255), 5)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1])

    return image



def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = MMDataParallel(model, device_ids=[0])

    width = 384
    height = 288
    c = np.array([320, 240])
    s = np.array([1.4, 1.8])
    r = 0
    image_size = [height, width]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    # camera 1
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while(True):
        ret, frame = cap.read()
        # rectangle
        cv2.rectangle(frame, (140, 13), (480, 473), (0, 255, 0), 2)

        trans = get_affine_transform(c, s, r, image_size)
        img_trans = cv2.warpAffine(frame, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        img_trans = F.to_tensor(img_trans)
        img_trans = F.normalize(img_trans, mean=mean, std=std)
        img_trans = torch.reshape(img_trans, (1, 3, width, height))

        data = {}
        data['img'] = img_trans
        data['img_metas'] = [{'image_file': '', 'center': c, 'scale': s, 'rotation': r, 'bbox_score': 1, 
                                'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], 'bbox_id': 0}]

        result = model(return_loss=False, **data)
        add_joints(frame, result['preds'][0])
        cv2.imshow('Lite HRNet', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
        
 
if __name__ == '__main__':
    main()
