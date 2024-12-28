import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    
    # ### FGADR
    # target_dir = 'D:/Med/dataset/DGDR/images/FGADR'
    # grades = ['nodr', 'mild_npdr', 'moderate_npdr', 'severe_npdr', 'pdr']
    # for g in grades:
    #     if os.path.exists(os.path.join(target_dir, g)) == False:
    #         os.mkdir(os.path.join(target_dir, g))
    # imgs_name = []
    # imgs_ann = []
    # with open('../splits/FGADR_train.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip('\n')
    #         imgs_name.append(line.split(' ')[0].split('/')[-1])
    #         imgs_ann.append(line.split(' ')[1])
    # with open('../splits/FGADR_crossval.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip('\n')
    #         imgs_name.append(line.split(' ')[0].split('/')[-1])
    #         imgs_ann.append(line.split(' ')[1])
    # origin_dir = 'D:/Med/dataset/DGDR/FGADR/processed'
    # for i, name in enumerate(imgs_name):
    #     origin_pth = os.path.join(origin_dir, name)
    #     new_pth = os.path.join(target_dir, grades[int(imgs_ann[i])], name)
    #     shutil.copyfile(origin_pth, new_pth)
        
        
    ### MESSIDOR
    target_dir = 'D:/Med/dataset/DGDR/images/MESSIDOR'
    grades = ['nodr', 'mild_npdr', 'moderate_npdr', 'severe_npdr', 'pdr']
    for g in grades:
        if os.path.exists(os.path.join(target_dir, g)) == False:
            os.mkdir(os.path.join(target_dir, g))
    imgs_name = []
    imgs_ann = []
    with open('D:/Med/dataset/DGDR/splits/MESSIDOR_train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            imgs_name.append(line.split(' ')[0].split('/')[-1])
            imgs_ann.append(line.split(' ')[1])
    with open('D:/Med/dataset/DGDR/splits/MESSIDOR_crossval.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            imgs_name.append(line.split(' ')[0].split('/')[-1])
            imgs_ann.append(line.split(' ')[1])
    origin_dir = 'D:/Med/dataset/DGDR/MESSIDOR2/processed'
    for i, name in enumerate(imgs_name):
        origin_pth = os.path.join(origin_dir, name)
        new_pth = os.path.join(target_dir, grades[int(imgs_ann[i])], name)
        shutil.copyfile(origin_pth, new_pth)