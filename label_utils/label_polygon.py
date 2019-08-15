import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from utils import *
import json
import pickle
from center import *
from tqdm import tqdm
import math

root = "../sbd"
instance_dir = os.path.join(root, "SegmentationObject/")
sem_dir = os.path.join(root, "SegmentationClass/")
label_dir = "../label_polygon_"

def compare_path(path_1, path_2, distMatrix):
    sum1 = 0
    for i in range(1, len(path_1)):
        sum1 += distMatrix[path_1[i-1]][path_1[i]]
    
    sum2 = 0
    for i in range(1, len(path_2)):
        sum2 += distMatrix[path_2[i-1]][path_2[i]]
    
    return sum1>sum2

def trans_polarone_to_another(ori_deg,assisPolar,center_coord,im_shape):
    '''
    make sure that the r,theta you want to assis not outof index
    assisPolar = (r,deg)
    center_coord = (center_x,center_y)
    '''
    assis_r = np.array(assisPolar[0],np.float32)
    ori_deg = np.array(ori_deg,np.float32)
    x = -1
    y = -1
    while not (x >= 0 and x < im_shape[1] and y >= 0 and y < im_shape[0]):
        x, y = cv.polarToCart(assis_r,ori_deg,angleInDegrees=True)
        x += center_coord[0]
        y += center_coord[1]
        x = int(x)
        y = int(y)
        ori_r = assis_r
        assis_r -= 0.1
    return ori_r



def fillInstance(instance, instance_id):
    _, contours, _= cv.findContours(instance, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        # whole instance
        # result in shape (N,1,2)

        # computing areas
        edgePoints = contours[0]
        for i in range(1, len(contours)):
            edgePoints = np.concatenate((edgePoints, contours[i]),axis=0)
        
        dictEdgePoint = {} # for later grouping
        for i in range(len(contours)):
            for j in range(contours[i].shape[0]):
                e_x = str(contours[i][j][0][0])
                e_y = str(contours[i][j][0][1])
                dictEdgePoint[e_x+"_"+e_y]=[i,j]
        
        # bbox of whole instance
        x, y, w, h = cv.boundingRect(edgePoints)

        # extract outline contour
        distanceMapUp = np.zeros((w+1,1))
        distanceMapUp.fill(np.inf)
        distanceMapDown = np.zeros((w+1,1))
        distanceMapDown.fill(-np.inf)
        distanceMapLeft = np.zeros((h+1, 1))
        distanceMapLeft.fill(np.inf)
        distanceMapRight = np.zeros((h+1, 1))
        distanceMapRight.fill(-np.inf)

        for edgePoint in edgePoints:
            p_x = edgePoint[0][0]
            p_y = edgePoint[0][1]
            index_x = p_x - x
            index_y = p_y - y
            if index_y < distanceMapUp[index_x]:
                distanceMapUp[index_x] = index_y
            if index_y > distanceMapDown[index_x]:
                distanceMapDown[index_x] = index_y
            if index_x < distanceMapLeft[index_y]:
                distanceMapLeft[index_y] = index_x
            if index_x > distanceMapRight[index_y]:
                distanceMapRight[index_y] = index_x
        
        # grouping outline to original contours, it can make undirected points partially directed
        selected_points = []
        selected_info = {}
        for i in range(w+1):
            if distanceMapUp[i] < np.inf:
                e_x = int(i+x)
                e_y = int(distanceMapUp[i]+y)
                selected_points.append([e_x, e_y])
                selected_info[str(e_x)+"_"+str(e_y)]=dictEdgePoint[str(e_x)+"_"+str(e_y)]
        for i in range(h+1):
            if distanceMapRight[i] > -np.inf:
                e_x = int(distanceMapRight[i]+x)
                e_y = int(i+y)
                selected_points.append([e_x, e_y])
                selected_info[str(e_x)+"_"+str(e_y)]=dictEdgePoint[str(e_x)+"_"+str(e_y)]
        for i in range(w,-1,-1):
            if distanceMapDown[i] > -np.inf:
                e_x = int(i+x)
                e_y = int(distanceMapDown[i]+y)
                selected_points.append([e_x, e_y])
                selected_info[str(e_x)+"_"+str(e_y)]=dictEdgePoint[str(e_x)+"_"+str(e_y)]
        for i in range(h,-1,-1):
            if distanceMapLeft[i] < np.inf:
                e_x = int(distanceMapLeft[i]+x)
                e_y = int(i+y)
                selected_points.append([e_x, e_y])
                selected_info[str(e_x)+"_"+str(e_y)]=dictEdgePoint[str(e_x)+"_"+str(e_y)]

        selected_info = sorted(selected_info.items(), key=lambda x:(x[1], x[0]))
        groups = {}
        for item in selected_info:
            name = item[0]
            coord_x = name.split("_")[0]
            coord_y = name.split("_")[1]
            c = item[1][0]
            try:
                groups[c].append((int(coord_x), int(coord_y)))
            except KeyError:
                groups[c] = [(int(coord_x), int(coord_y))]

        # connect group
        start_list = []
        end_list = []
        point_number_list = []
        for key in groups.keys():
            # inside each group, shift the array, so that the first and last point have biggest distance
            tempGroup = groups[key].copy()
            tempGroup.append(tempGroup.pop(0))
            distGroup = np.diag(distance.cdist(groups[key], tempGroup, 'euclidean'))
            max_index = np.argmax(distGroup)
            if max_index != len(groups[key])-1:
                groups[key] = groups[key][max_index+1:]+groups[key][:max_index+1]
            point_number_list.append(len(groups[key]))
            start_list.append(groups[key][0])
            end_list.append(groups[key][-1])
        
        # get center point here
        point_count = 0
        center_x = 0
        center_y = 0
        for i in range(len(start_list)):
            center_x += start_list[i][0]
            center_x += end_list[i][0]
            center_y += start_list[i][1]
            center_y += end_list[i][1]
            point_count += 2
        center_x /= point_count
        center_y /= point_count

        # calculate the degree based on center point
        degStartList = []
        for i in range(len(start_list)):
            deg = -np.arctan2(1,0) + np.arctan2(start_list[i][0]-center_x,start_list[i][1]-center_y)
            deg = deg * 180 / np.pi
            if deg < 0:
                deg += 360
            degStartList.append(deg)
        
        # first solely consider the degree, construct a base solution
        best_path = np.argsort(degStartList)
        best_path = np.append(best_path, best_path[0])

        
        # then consider distance, model it as asymmetric travelling salesman problem
        # note: add this step the solution is not necessarily better
        # note: if an object is relatively simple, i.e. <=3 area, do not need this
        # TODO: find a more robust solution here
        if len(groups.keys())>4:
            distMatrix = distance.cdist(end_list, start_list, 'euclidean')

            MAX_ITER = 100
            count = 0
            while count < MAX_ITER:
                path = best_path.copy()
                start = np.random.randint(1, len(path)-1)
                if np.random.random() > 0.5:
                    while start-2 <= 1:
                        start = np.random.randint(1, len(path)-1)
                    end = np.random.randint(1, start-2)
                    path[end: start+1] = path[end: start+1][::-1]
                else:
                    while start+2 >= len(path) -1:
                        start = np.random.randint(1, len(path)-1)
                    end = np.random.randint(start+2, len(path)-1)
                    path[start: end+1] = path[start: end+1][::-1]
                if compare_path(best_path, path, distMatrix):
                    count = 0
                    best_path = path
                else:
                    count+=1
        final_points = []
        groupList= list(groups.keys())
        for i in range(len(best_path)-1):
            final_points += groups[groupList[best_path[i]]]
        final_points = np.array(final_points)

        # fill the break piece
        cv.fillPoly(instance, [final_points], (int(instance_id),0,0))
    return instance

# input instance with only one contour
def getOrientedPoints(instance):    
    # first get center point
    instance = instance.astype(np.uint8)
    center_x, center_y= centerdot(instance) # your implementation, return a tuple or a list center = (center_x, center_y)
    
    edges = get_gradient(instance) # your implementation of get gradient, it is a bool map
    index_h, index_w = np.where(edges == 1)
    edgepoints_array = np.array([(index_w[i], index_h[i]) for i in range(len(index_h))])  # x, y
    centerpoints_array = np.array([center_x, center_y])
    # distance_all = distance.cdist(edgepoints_array,centerpoints_array,'euclidean')

    edgeDict = {} # we create a dict for which key = 0, 1,2,3,...359 value list of distance
    # generate empty list for all the angle
    for i in range(360):
        edgeDict[str(i)] = []
    for i in range(len(index_h)):
        # # calculate the degree based on center point 
        # clockwise
        # i want to get a deg section of each points
        deg_1 = -np.arctan2(1,0) + np.arctan2(index_w[i]-center_x,index_h[i]-center_y)
        deg_1 = deg_1 * 180 / np.pi
        if deg_1 < 0:
            deg_1 += 360
        deg_2 = -np.arctan2(1,0) + np.arctan2(index_w[i]+1-center_x,index_h[i]+1-center_y)
        deg_2 = deg_2 * 180 / np.pi
        if deg_2 < 0:
            deg_2 += 360
        deg_3 = -np.arctan2(1,0) + np.arctan2(index_w[i]-center_x,index_h[i]+1-center_y)
        deg_3 = deg_3 * 180 / np.pi
        if deg_3 < 0:
            deg_3 += 360
        deg_4 = -np.arctan2(1,0) + np.arctan2(index_w[i]+1-center_x,index_h[i]-center_y)
        deg_4 = deg_4 * 180 / np.pi
        if deg_4 < 0:
            deg_4 += 360
        deg1 = min(deg_1,deg_2,deg_3,deg_4)
        deg2 = max(deg_1,deg_2,deg_3,deg_4)
        # calculate distance
        dot_array = np.array([index_w[i],index_h[i]])
        distance_r = np.linalg.norm(dot_array - centerpoints_array)
        # consider when deg = 0 
        if int(deg2 - deg1) > 100:
            for deg in range(0,math.ceil(deg1)):
                edgeDict[str(int(deg))].append(distance_r)
            for deg in range(math.ceil(deg2),360):
                edgeDict[str(int(deg))].append(distance_r)
        else:
            for deg in range(math.ceil(deg1),math.ceil(deg2)):
                edgeDict[str(int(deg))].append(distance_r)
        
    # sorted method
    # edgeDict = {k:sorted(edgeDict[k]) for k in edgeDict.keys()}
    start_deg = 0
    '''
    change start_points
    '''
    # find the largest r for each deg
    try:
        edgeDict = {k:np.max(np.array(edgeDict[k])) for k in edgeDict.keys()}
    except ValueError:
        for index_deg in range(360):
            if len(edgeDict[str(index_deg)]) == 0:
                search_deg = index_deg
                while len(edgeDict[str(search_deg%360)]) == 0:
                    search_deg += 1
                search_info = edgeDict[str(search_deg%360)]

                for r_info in search_info:
                    assisPolar = (search_deg%360,r_info)
                    center_coord  =(center_x,center_y )
                    trans_r = trans_polarone_to_another(index_deg,assisPolar,center_coord,instance.shape)
                    edgeDict[str(index_deg)].append(trans_r)
        edgeDict = {k:np.max(np.array(edgeDict[k])) for k in edgeDict.keys()}
    points = [edgeDict[str(deg_num)] for deg_num in range(360)] # start 0 deg 
    
    return points,center_x,center_y

def getPointsOfNumber(edgePoints, N):
    while len(edgePoints) < N: # if origin edgePoint is less than N
        tempEdge = edgePoints.copy().tolist()
        tempEdge.append(tempEdge.pop(0)) # rotate the edgePoints
        distGroup = np.diag(distance.cdist(edgePoints, tempEdge, 'euclidean'))
        index = np.argmax(distGroup) # find the largest differential
        # insert middle point from index to index+1
        if index+1 < len(edgePoints):
            middle_points = (edgePoints[index] + edgePoints[index+1])/2
        else:
            middle_points = (edgePoints[index] + edgePoints[1])/2
        edgePoints = edgePoints.tolist()
        edgePoints.insert(index+1,middle_points)
        edgePoints = np.array(edgePoints)

    if len(edgePoints) == N:
        return edgePoints
    else: # need reduction
        # by using opencv approxPolyDP
        error_tolerance = 0.001
        epsilon = 0.001*cv.arcLength(edgePoints, True)
        approx = cv.approxPolyDP(edgePoints, epsilon, True)
        while len(approx) > N:
            error_tolerance *= 1.1
            epsilon = error_tolerance*cv.arcLength(approx, True)
            approx = cv.approxPolyDP(approx, epsilon, True)
            #print(len(approx) > N)
        while len(approx) < N:
            tempEdge = approx[:,0,:].copy().tolist()
            tempEdge.append(tempEdge.pop(0)) # rotate the edgePoints
            distGroup = np.diag(distance.cdist(approx[:,0,:], tempEdge, 'euclidean'))
            index = np.argmax(distGroup) # find the largest differential
            # insert middle point from index to index+1
            if index+1 < len(approx):
                middle_points = (approx[index] + approx[index+1])/2
            else:
                middle_points = (approx[index] + approx[1])/2
            
            approx = approx.tolist()
            approx.insert(int(index+1),middle_points)
            approx = np.array(approx)
        approx.resize(len(approx),2)
        return approx

def getMaxAreaContour(contours):
    if len(contours)>1:
        areas = []
        for contour in contours:
            areas.append(cv.contourArea(contour))
        sorted_index = np.argsort(areas)
        edgePoints = contours[sorted_index[-1]]
    else:
        edgePoints = contours[0]
    return edgePoints

def runOneImage(img_path,save_dir,polygon_num):
    instance_mask = Image.open(img_path) # PIL
    instance_mask = np.array(instance_mask)
    instance_ids = np.unique(instance_mask)
    semantic_mask = np.array(Image.open(img_path.replace("Object", "Class")))
    img_name = img_path.split('/')[-1]
 
    label_dir_pkl = os.path.join(save_dir,'label_pkl')
    label_dir_txt = os.path.join(save_dir,'label_txt')
    if not os.path.exists(label_dir_pkl):
        os.mkdir(label_dir_pkl)
    if not os.path.exists(label_dir_txt):
        os.mkdir(label_dir_txt)   
    imw = instance_mask.shape[1]
    imh = instance_mask.shape[0]
    img_info_dict = []
    has_object = False
    for instance_id in instance_ids:
        objects_info = {}
        if instance_id==0 or instance_id==255: # background or edge, pass
            continue
        # extract instance
        temp = np.zeros(instance_mask.shape)
        temp.fill(instance_id)
        tempMask = (instance_mask == temp)
        cat_id = np.max(np.unique(semantic_mask * tempMask)) # semantic category of this instance
        instance = instance_mask * tempMask
        instance_temp = instance.copy() # findContours will change instance, so copy first
        instance = fillInstance(instance_temp, instance_id)
        _, contours, _= cv.findContours(instance, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        edgePoints = getMaxAreaContour(contours)

        instance_second = np.zeros(instance_mask.shape)
        cv.fillPoly(instance_second, [edgePoints], (int(instance_id),0,0))
        
        x,y,w,h = cv.boundingRect(instance_second.astype(np.uint8))
        x += w/2
        y += h/2 
        point_number = polygon_num
        try:
            points = getPointsOfNumber(edgePoints[:,0,:], point_number) #x,y
        except:
            print('invaild label')
            continue
        has_object = True
        objects_info['label'] = cat_id
        objects_info['imgwh'] = (imw,imh)
        objects_info['bbox'] = (x,y,w,h)
        objects_info['polygon_info'] = points
        img_info_dict.append(objects_info)
    if has_object == True:
        with open(os.path.join(label_dir_pkl,img_name[:-4]+'.pkl'),'wb') as fpkl:
            pickle.dump(img_info_dict,fpkl)
        info_txt = np.zeros((len(img_info_dict),2*point_number+7))
        for i in range(len(img_info_dict)):
            info_txt[i][0] = img_info_dict[i]['label']
            info_txt[i][1:3] = img_info_dict[i]['imgwh']
            info_txt[i][3:7] = img_info_dict[i]['bbox']
            a = img_info_dict[i]['polygon_info']
            a = a.reshape(2*point_number)
            a = a.reshape(point_number,2).T
            x_coord = a[0]
            y_coord = a[1]
            info_txt[i][7:7+point_number] = x_coord
            info_txt[i][7+point_number:7+2*point_number] = y_coord
        np.savetxt(os.path.join(label_dir_txt,img_name[:-4]+'.txt'),info_txt)
                

if __name__ == "__main__":
    inst_list = os.listdir(instance_dir)
    #for poly_num in tqdm(range(1,11)):
    poly_num = 360
    label_save_dir = label_dir+str(poly_num)
    if not os.path.exists(label_save_dir):
        os.mkdir(label_save_dir)
    label_dir_pkl = os.path.join(label_save_dir,'label_pkl')
    label_dir_txt = os.path.join(label_save_dir,'label_txt')
    if not os.path.exists(label_dir_pkl):
        os.mkdir(label_dir_pkl)
    if not os.path.exists(label_dir_txt):
        os.mkdir(label_dir_txt)

    for i in tqdm(range(len(inst_list))):
        runOneImage(os.path.join(instance_dir ,inst_list[i]), label_save_dir, poly_num)
        
