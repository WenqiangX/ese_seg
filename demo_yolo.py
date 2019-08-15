"""YOLO Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
import numpy as np
import time
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2 as cv
import shutil
from matplotlib.ticker import NullLocator
def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                        help="Base network name yolo3_darknet53_voc\yolo3_tiny_darknet_voc")
    parser.add_argument('--images', type=str, default= '',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.45,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    if not os.path.exists(args.images):
        os.mkdir(args.images)  
    # grab some image if not specified
    if not args.images.strip():
        gcv.utils.download("https://cloud.githubusercontent.com/assets/3307514/" +
            "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg", 'street.jpg')
        image_list = ['street.jpg']
    else:
        image_nameList = os.listdir(args.images)
        image_list = []
        for i in image_nameList:
            image_list.append(os.path.join(args.images,i))
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
        net.load_parameters(args.pretrained)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        
    image_list_batch = image_list 
    total_time = 0
    for image in tqdm(image_list_batch):
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1)
        img_str = image.split('/')[-1]
        x, img = presets.yolo.load_test(image, short=416)
        img_w = img.shape[0]
        img_h = img.shape[1]
        x = x.as_in_context(ctx[0])
        a = time.time()
        ids, scores, bboxes, absolute_coef_centers, coef = [xx[0].asnumpy() for xx in net(x)]
        ax = gcv.utils.viz.plot_r_polygon(img, bboxes,absolute_coef_centers, coef, img_w, img_h, scores, ids , thresh=args.thresh,class_names=net.classes, ax=ax)
        b = time.time()
        each_time = b-a
        total_time += each_time
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.subplots_adjust(top = 0.995, bottom = 0.005, right = 0.995, left = 0.005, hspace = 0, wspace = 0)
        plt.savefig(os.path.join(args.save_dir,img_str))
        plt.close()
    print("network speed ",1.0*len(image_list_batch) / total_time, "fps")
    