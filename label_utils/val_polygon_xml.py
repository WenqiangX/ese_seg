# generate polygon label txt to polygon label xml
# polygon label format
#    size (N, 727)
#    cat_id, img_w, img_h, bbox_x, bbox_y, bbox_w, bbox_h, polygon_x(360), polygon_y(360)
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
#from skimage.segmentation import active_contour
#from scipy.spatial import distance
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from tqdm import tqdm

# Pascal VOC 20 class label name
labels=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def save_xml(img_name, cat_list, img_info, save_dir, width, height, channel):
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
    for obj_info in img_info:
        bbox_center_x, bbox_center_y = obj_info[3], obj_info[4]
        bbox_w, bbox_h = obj_info[5], obj_info[6]
        bbox_xmin,bbox_ymin = bbox_center_x - bbox_w / 2.0, bbox_center_y - bbox_h / 2.0
        bbox_xmax,bbox_ymax = bbox_center_x + bbox_w / 2.0, bbox_center_y + bbox_h / 2.0
        
        points_x = obj_info[7:7+360]
        points_y = obj_info[7+360:7+2*360]
        points_x_str = str(points_x)
        points_y_str = str(points_y)
        # Delete '[', ']'
        points_x_str = points_x_str[1:-1]
        points_y_str = points_y_str[1:-1]
        
        #############################################
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
        
        node_polygon_x = SubElement(node_object, 'points_x')
        node_polygon_x.text = '%s' % points_x_str
        node_polygon_y = SubElement(node_object, 'points_y')
        node_polygon_y.text = '%s' % points_y_str
        ############################################
        count += 1
        
    xml = tostring(node_root, pretty_print=True)  
    dom = parseString(xml)
    '''
    write the label list
    with open('./train_8_xy_bboxwh.txt','a') as f:
        f.write(img_name+'\n')
    '''
    save_xml = os.path.join(save_dir, img_name+'.xml')
    with open(save_xml, 'wb') as f:
        f.write(xml)

def extractTXT(txt_path):
    '''
    read polygon label_txt
    
    return img_name, cat_list, img_info, width, height, channel 
    '''
    txt_name = txt_path.split('/')[-1][:-4]
    with open(txt_path,'r') as f:
        img_info = np.loadtxt(f)
        img_info = img_info.reshape((-1,7+360*2))    # 360 is the edge points number
        '''
        img_info:
        Size (N, 7+360*2)
        cat_id, img_w, img_h, bbox_x, bbox_y, bbox_w, bbox_h, polygon_x(360), polygon_y(360)
        '''
    img_name = txt_name
    cat_list = []    # Cat list in one img
    for i in range(len(img_info)):
        cat_id = int(img_info[i][0])
        cat_list.append(cat_id)
    width = img_info[0][1]
    height = img_info[0][2]
    channel = 3

    return img_name, cat_list, img_info, width, height, channel

if __name__ == '__main__':
    txt_dir = '../label_polygon_360/label_txt/'
    save_dir = '../sbd/label_polygon_360_xml/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    txt_list = os.listdir(txt_dir)
    for txt_file in tqdm(txt_list):
        txt_path = os.path.join(txt_dir,txt_file)
        img_name, cat_list, img_info, width, height, channel = extractTXT(txt_path)
        save_xml(img_name, cat_list, img_info, save_dir, width, height, channel)
