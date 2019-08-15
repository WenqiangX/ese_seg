"""Train YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.voc_polygon_detection import VOC07PolygonMApMetric
from gluoncv.utils import LRScheduler
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='tiny_darknet',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=2, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',#'./yolo3_darknet53_voc_0000_0.6948.params',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=240,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default= '160,180',
                        help='epochs at which learning rate decays. default is 260,280.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='./result_test/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=10,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_false',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                        'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    parser.add_argument('--deg', type=int, default=8, help='cheby degree , actually 17')
    parser.add_argument('--val_voc2012', type=bool, default=False, help='val in pascal voc 2012')
    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        if args.val_voc2012:
            val_dataset = gdata.VOC_Val_Detection(
            splits=[('sbdche', 'val_2012_bboxwh')])
        else:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val'+'_'+str(args.deg)+'_bboxwh')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        val_polygon_metric = VOC07PolygonMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric, val_polygon_metric

def get_dataloader(net, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return val_loader

def validate(net, val_data, ctx, eval_metric, polygon_metric, args):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in tqdm(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        det_coef_centers = []
        det_coefs = []
        det_r_all = []
        gt_bboxes = []
        gt_points_xs = []
        gt_points_ys = []
        gt_ids = []
        gt_difficults = []
        gt_widths = []
        gt_heights = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes, absolute_coef_centers, coef = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_coef_centers.append(absolute_coef_centers)
            det_coefs.append(coef)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4+720, end=5+720))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_points_xs.append(y.slice_axis(axis=-1, begin=4, end=4+360))
            gt_points_ys.append(y.slice_axis(axis=-1, begin=4+360, end=4+720))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5+720, end=6+720) if y.shape[-1] > 5 else None)
            gt_widths.append(y.slice_axis(axis=-1, begin=6+720, end=7+720))
            gt_heights.append(y.slice_axis(axis=-1, begin=7+720, end=8+720))
        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        polygon_metric.update(det_bboxes, det_coef_centers, det_coefs, det_ids, det_scores, gt_bboxes, gt_points_xs, gt_points_ys, gt_ids, gt_widths, gt_heights, gt_difficults)
    return eval_metric.get(), polygon_metric.get()
def demo_val(net, val_data, eval_metric, polygon_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True
        
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    tic = time.time()
    btic = time.time()
    mx.nd.waitall()
    net.hybridize()

    map_bbox, map_polygon = validate(net, val_data, ctx, eval_metric, polygon_metric,args)
    map_name, mean_ap = map_bbox
    polygonmap_name, polygonmean_ap = map_polygon
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    logger.info('[Epoch {}] Validation: \n{}'.format(0, val_msg))
    polygonval_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(polygonmap_name, polygonmean_ap)])
    logger.info('[Epoch {}] PolygonValidation: \n{}'.format(0, polygonval_msg))

if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo3', args.network, args.dataset))
    args.save_prefix += net_name
    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
    else:
        net = get_model(net_name, pretrained_base=True)
        async_net = net
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
    print(net)
    # val data
    val_dataset, eval_metric, polygon_metric= get_dataset(args.dataset, args)
    val_data = get_dataloader(
        async_net, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)
    ################
    # Valing
    demo_val(net, val_data, eval_metric, polygon_metric, ctx, args)
    
