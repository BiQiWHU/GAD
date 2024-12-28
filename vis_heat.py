import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
from matplotlib import cm
import random

import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE


def read_split(split_file, img_root):
    items = []
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            impath, label = line.split(" ")
            impath = os.path.join(img_root, impath)
            label = int(label)
            items.append((impath, label))
            
    return items


if __name__ == '__main__':
    
    # feature = np.load('feature.npy')
    # roots =['fcitys_bad', 'facdc_bad', 'fzur_bad', 'fdri_bad']
    # status = ['good', 'bad']
    domains = ['APTOS', 'DEEPDR', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR']
    # stages = ['stage1', 'stage2', 'stage3', 'stage4', 'preds']
    # stages = ['preds']
    img_root = 'E:/Med/dataset/DGDR/images'
    heat_root = 'E:/Med/DGDR/vis/heat_/'
    feat_root = 'D:/Med/DGDR/heat/l2'
    split_dir = 'E:/Med/dataset/DGDR/splits'
    
    for i, d in enumerate(domains):
        file_train = os.path.join(split_dir, d + "_train.txt")
        impath_label_list = read_split(file_train, img_root)
        file_val = os.path.join(split_dir, d + "_crossval.txt")
        impath_label_list += read_split(file_val, img_root)
        img_pths = []
        img_names = []
        feat = np.array(np.load('{}/heat_{}.npy'.format(feat_root, d), allow_pickle=True))
        heat_cls = os.path.join(heat_root, d)
        if os.path.exists(heat_cls) == False:
            os.mkdir(heat_cls)
        ind = 0
        for cls_dir in tqdm(os.listdir(os.path.join(img_root, d))):
            cls_count = 0
            img_dir = os.path.join(img_root, d, cls_dir)
            heat_dir = os.path.join(heat_root, d, cls_dir)
            if os.path.exists(heat_dir) == False:
                os.mkdir(heat_dir)   
            for img in os.listdir(img_dir):
                img_pths.append(os.path.join(img_dir, img))
                img_names.append(img)
                img = cv2.imread(img_pths[ind])
                name = img_names[ind]
                att_feat = feat[ind]
                # h = att_feat.shape[0]
                # att_feat[0, 0] = 0
                # att_feat[0, h-1] = 0
                # att_feat[h-1, 0] = 0
                # att_feat[h-1, h-1] = 0
                if cls_count < 20:
                    amin, amax = att_feat[:, :].min(), att_feat[:, :].max()
                    attention_map1 = 255 * (att_feat[:, :] - amin) / (amax - amin)
                    attention_map1[attention_map1 < 230] *= 0.6
                    attention_map1[attention_map1 > 230] *= 1.2
                    attention_map1[attention_map1 > 255] = 255
                    attention_map1 = attention_map1.astype(np.uint8)
                    attention_map1 = cv2.resize(attention_map1, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
                    attention_map = attention_map1.astype(np.uint8)
                    attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
                    img_add = cv2.addWeighted(img, 0.8, attention_map, 0.2, 0)
                    cv2.imwrite(os.path.join(heat_dir, name), img_add)
                ind += 1
                cls_count += 1
