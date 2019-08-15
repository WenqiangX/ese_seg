import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import argparse
import math
import random

root_dir = os.getcwd()



#input im hxw     im[:,:,0]
#add 2
def add_edge(im):
    h,w = im.shape[0],im.shape[1]
    #add_edge_im = np.zeros((h+10,w+10))
    #add_edge_im[5:h+5,5:w+5] = im
    add_edge_im = im
    return add_edge_im 

def get_gradient(im):
    h,w = im.shape[0],im.shape[1]
    im  = add_edge(im)
    instance_id = np.unique(im)[1]
    #delete line
    mask = np.zeros((im.shape[0],im.shape[1]))
    mask.fill(instance_id)
    boolmask = (im == mask)
    im = im * boolmask    #only has object

    y = np.gradient(im)[0]
    x = np.gradient(im)[1]
    gradient = abs(x)+abs(y)
    bool_gradient= gradient.astype(bool)
    mask.fill(1)
    gradient_map = mask*bool_gradient*boolmask
    #gradient_map = gradient_map[5:h+5,5:w+5]
    #2d gradient map
    return  gradient_map

def get_startpoint(gradient_map,direction = 0,bias=0):
    '''
    v2
    '''
    h,w = gradient_map.shape[0],gradient_map.shape[1]
    Find_start_point = False
    h_location,w_location = np.nonzero(gradient_map)[0],np.nonzero(gradient_map)[1]
    h_center = int(np.sum(h_location)/h_location.shape[0])
    w_center = int(np.sum(w_location)/w_location.shape[0])
    #scan height
    if direction == 0:
        while not Find_start_point:
            for index_h in range(h):
                if gradient_map[index_h][w_center+bias] != 0:
                    startpoint = (index_h,w_center)
                    Find_start_point = True
                    break
            bias += 1
    elif direction == 1 or Find_start_point == False:
        for index_w in range(w):
            if gradient_map[h_center+bias][index_w] != 0:
                startpoint = (h_center,index_w)
                Find_start_point = True
                break
    return startpoint

def serach_box(search_map,index_h,index_w,scale):
    present_box = search_map[index_h-scale:index_h+scale+1,index_w -scale:index_w+scale+1]
    present_box[scale][scale] = 0
    next_index = np.nonzero(present_box)
    return next_index

def get_boundingorder(gradient_map,direction=0): #im_name is used to visualize
    #clockwise
    #index out
    search_map = gradient_map
    orderlist = []
    (start_pointh,start_pointw) = get_startpoint(search_map,direction = direction)
    start_point = (start_pointh,start_pointw)
    present_point = (start_pointh,start_pointw)
    orderlist.append(present_point)
    #second point
    bias = 0
    while present_point == start_point:
        if search_map[start_pointh][start_pointw+1] == 1:
            #the points have been searched are deleted 
            search_map[present_point[0]][present_point[1]] = 0
            present_point = (start_pointh,start_pointw+1)
            orderlist.append(present_point)

        elif search_map[start_pointh+1][start_pointw+1] == 1:
            search_map[present_point[0]][present_point[1]] = 0
            present_point = (start_pointh+1,start_pointw+1)
            orderlist.append(present_point)

        elif search_map[start_pointh+1][start_pointw] == 1:
            search_map[present_point[0]][present_point[1]] = 0
            present_point = (start_pointh+1,start_pointw)
            orderlist.append(present_point)
        else:
            bias += 1
            (start_pointh,start_pointw) = get_startpoint(search_map,direction = direction, bias = bias)
            start_point = (start_pointh,start_pointw)

    point_num = 2
    while present_point != start_point:
        index_h ,index_w = present_point[0], present_point[1]
        scale = 1
        next_index = serach_box(search_map,index_h,index_w,scale)
        while next_index[0].shape[0] == 0 or next_index[1].shape[0] == 0:
            scale += 1
            next_index = serach_box(search_map,index_h,index_w,scale)
        
        next_index_h,next_index_w = next_index[0][0]-1+index_h , next_index[1][0]-1+index_w
        present_point = (next_index_h,next_index_w)
        orderlist.append(present_point)
        point_num += 1
        if point_num == 4:
            search_map[start_pointh][start_pointw] = 1
    return orderlist

def get_polygonmap_n(boundingorder,N,im_name):
    boundlen = len(boundingorder)
    if N > boundlen:
        result = []
        return result
    point = []
    cut_interval = int(boundlen/N)
    for i in range(N):
        location = i * cut_interval
        point.append(boundingorder[location])
    return point

def get_polygon_area(points):
    area = 0
    if(len(points)<3):       
         raise Exception("error")
    p1 = points[0]
    for i in range(1,len(points)-1):
        p2 = points[i]
        p3 = points[i+1]
        #cacula vector
        vecp1p2 = (p2[0]-p1[0],p2[1]-p1[1])
        vecp2p3 = (p3[0]-p2[0],p3[1]-p2[1])
        #wheather clock
        vecMult = vecp1p2[0]*vecp2p3[1] - vecp1p2[1]*vecp2p3[0]  
        sign = 0
        if(vecMult>0):
            sign = 1
        elif(vecMult<0):
            sign = -1

        triArea = GetAreaOfTriangle(p1,p2,p3)*sign
        area += triArea
    return abs(area)

def GetAreaOfTriangle(p1,p2,p3):
    '''triangle area   Heron's formula'''
    area = 0
    p1p2 = GetLineLength(p1,p2)
    p2p3 = GetLineLength(p2,p3)
    p3p1 = GetLineLength(p3,p1)
    s = (p1p2 + p2p3 + p3p1)/2
    area = s*(s-p1p2)*(s-p2p3)*(s-p3p1)   #海伦公式
    area = math.sqrt(area)
    return area

def GetLineLength(p1,p2):
    '''caclulate length'''
    length = math.pow((p1[0]-p2[0]),2) + math.pow((p1[1]-p2[1]),2)  #pow  次方
    length = math.sqrt(length)   
    return length    

def get_mask_area(im):
    im  = add_edge(im)
    #delete line
    mask = np.zeros((im.shape[0],im.shape[1]))
    mask.fill(38)
    boolmask = (im == mask)
    area = np.sum(boolmask)
    return area
def link_line(start,end,polygon_map):
    #line points
    #input two points 
    #return a line link two points
    polygon_map[start[0]][start[1]] = 1
    polygon_map[end[0]][end[1]] = 1
    h_length = end[0]-start[0]
    w_length = end[1]-start[1]
    if h_length == 0:
        h_direction = 0
    else:
        h_direction = int((h_length)/abs(h_length))
    if w_length == 0:
        w_direction = 0
    else:
        w_direction = int((w_length)/abs(w_length))
    if (abs(h_length) > abs(w_length)):
        difference = abs(h_length) - abs(w_length)
        move_times = abs(h_length)
        move = [] #document if to skip   0 means h+(-1,1),w+(1,-1)
        if difference == 0:
            move = np.zeros((move_times))
            move = move.tolist()
        else:
            #interval = int(abs(h_length)/difference)
            move = np.zeros((move_times))
            while np.sum(move) < difference:
                a = random.randint(0,move.shape[0]-1)
                move[a] = 1
            move = move.tolist()
        index_h,index_w = start[0],start[1]
        for i in range(move_times):
            if move[i] == 0:
                index_h += h_direction
                index_w += w_direction
                polygon_map[index_h][index_w] = 1
            else:
                index_h += h_direction
                polygon_map[index_h][index_w] = 1
    
    if (abs(w_length) > abs(h_length)):
        difference = abs(w_length) - abs(h_length)
        move_times = abs(w_length)
        move = [] #document if to skip   0 means h+(-1,1),w+(1,-1)
        if difference == 0:
            move = np.zeros((move_times))
            move = move.tolist()
        else:
            #interval = int(abs(h_length)/difference)
            move = np.zeros((move_times))
            while np.sum(move) < difference:
                a = random.randint(0,move.shape[0]-1)
                move[a] = 1
            move = move.tolist()
        index_h,index_w = start[0],start[1]
        for i in range(move_times):
            if move[i] == 0:
                index_h += h_direction
                index_w += w_direction
                polygon_map[index_h][index_w] = 1
            else:
                index_w += w_direction
                polygon_map[index_h][index_w] = 1
    return polygon_map


def get_polygon_map(points,gradient_map):
    polygon_map = np.zeros((gradient_map.shape[0],gradient_map.shape[1]))
    #link line to polygon
    for i in range(len(points)):
        if i < (len(points)-1):
            point1 = points[i]
            point2 = points[i+1]
        else:
            point1 = points[i]
            point2 = points[0]
        polygon_map = link_line(point1,point2,polygon_map)
    #fill the polygon
    #from up to down ,from left to right
    for w_index in range(polygon_map.shape[1]):
        point_pairs = []#strat point,endpoint or only one point
        for h_index in range(polygon_map.shape[0]):
            if (polygon_map[h_index][w_index] != 0) and (len(point_pairs) == 0):
                point_pairs.append((h_index,w_index))
            elif (polygon_map[h_index][w_index] != 0) and (len(point_pairs) == 1):
                point_pairs.append((h_index,w_index))
                polygon_map = link_line(point_pairs[0],point_pairs[1],polygon_map)
                point_pairs = []
                point_pairs.append((h_index,w_index))
            else:
                continue
    return polygon_map
