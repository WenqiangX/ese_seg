import numpy as np
from tqdm import tqdm
import numpy.polynomial.chebyshev  as chebyshev
import os

def che_fit(txt_path,deg,fit_save_dir):
    '''
    input : a txt of    label, imgw, imgh, x, y, w, h, centerx, centery, 360deg 
    return label, imgw, imgh, x, y, w, h, centerx, centery, coef(16,24)
    '''
    with open(txt_path,'r') as f:
        img_info = np.loadtxt(txt_path)
        img_info = img_info.reshape(-1,369)
        new_path = os.path.join(fit_save_dir,txt_path.split('/')[-1])

        results = []
        for objects_info in img_info:
            # 1,360
            objects_new = np.zeros(9+deg+1)
            objects_new[0:9]= objects_info[0:9]
            bboxw = objects_info[5]
            bboxh = objects_info[6]
            bbox_len = np.sqrt(bboxw*bboxw+bboxh*bboxh)
            r = objects_info[9:] / float(bbox_len)
            theta = np.linspace(-1,1,360)
            coefficient,Res = chebyshev.chebfit(theta,r,deg,full=True)
            objects_new[9:] = np.array(coefficient)
            results.append(objects_new)
        results = np.array(results)
        np.savetxt(new_path, results)


if __name__ == '__main__':
    
    deg = 8
    center360_dir = '../label_center_edage/label_txt'
    if not os.path.exists("../sbd/cheby_fit"):
        os.mkdir("../cheby_fit")
    deg_dir = "../cheby_fit/n"+str(deg)
    if not os.path.exists(deg_dir):
        os.mkdir(deg_dir)
    fit_save_dir = os.path.join(deg_dir,'txt')
    if not os.path.exists(fit_save_dir):
        os.mkdir(fit_save_dir)
    center_list = os.listdir(center360_dir)
    print('fitting sbd')
    for txtname in tqdm(center_list):
        txt_path = os.path.join(center360_dir,txtname)
        che_fit(txt_path, 2*deg+1, fit_save_dir)
    print('fitting end')
