import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from tqdm import tqdm

labels=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
deg = 8
root = '../sbd/cheby_fit/'
if not os.path.exists(root):
    os.mkdir(root)
def save_xml(img_name, cat_list, pointsList, save_dir, width, height, channel):
    has_objects = False
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    count = 0
    for points in pointsList:
        coef = points[9:]
        bbox_center_x, bbox_center_y = points[3], points[4]
        bbox_w, bbox_h = points[5], points[6]
        bbox_xmin,bbox_ymin = bbox_center_x - bbox_w / 2.0, bbox_center_y - bbox_h / 2.0
        bbox_xmax,bbox_ymax = bbox_center_x + bbox_w / 2.0, bbox_center_y + bbox_h / 2.0
        coef_center_x ,coef_center_y = points[7], points[8]
       
        coef = points[9:]
        coef_str = str(points[9:])
        coef_str = coef_str[1:-1]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = labels[cat_list[count]-1]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % bbox_xmin
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % bbox_ymin
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % bbox_xmax
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bbox_ymax
        node_coef_center_x = SubElement(node_bndbox, 'coef_center_x')
        node_coef_center_x.text = '%s' % coef_center_x
        node_coef_center_y = SubElement(node_bndbox, 'coef_center_y')
        node_coef_center_y.text = '%s' % coef_center_y
        node_polygon = SubElement(node_object, 'coef')
        node_polygon.text = '%s' % coef_str
        count+=1
        has_objects = True
    xml = tostring(node_root, pretty_print=True)  
    dom = parseString(xml)

    if has_objects:
        with open(os.path.join(root,'coef_8_success.txt'),'a') as f:
            f.write(img_name+'\n')
    save_xml = os.path.join(save_dir, img_name+'.xml')
    with open(save_xml, 'wb') as f:
        f.write(xml)
        
def extractTXT(txt_path):
    txt_name = txt_path.split('/')[-1][:-4]
    with open(txt_path,'r') as f:
        img_info = np.loadtxt(f)
        img_info = img_info.reshape((-1,9+2*deg+2))
    img_name = txt_name
    cat_list = []
    for i in range(len(img_info)):
        cat_id = int(img_info[i][0])
        cat_list.append(cat_id)

    points_list = img_info
    width = img_info[0][1]
    height = img_info[0][2]
    channel = 3

    return img_name, cat_list, points_list, width, height, channel

if __name__ == '__main__':
    txt_dir = os.path.join("../cheby_fit", 'n'+str(deg), 'txt')
    save_dir = os.path.join(root, 'n'+str(deg)+'_xml')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    txt_list = os.listdir(txt_dir)
    for txt_file in tqdm(txt_list):
        txt_path = os.path.join(txt_dir,txt_file)
        img_name,cat_list,pointsList,width,height,channel = extractTXT(txt_path)
        save_xml(img_name, cat_list, pointsList, save_dir, width, height, channel)

