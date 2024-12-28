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



if __name__ == '__main__':
    
    # feature = np.load('feature.npy')
    # roots =['fcitys_bad', 'facdc_bad', 'fzur_bad', 'fdri_bad']
    # status = ['good', 'bad']
    domains = ['APTOS', 'DEEPDR', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR']
    # stages = ['stage1', 'stage2', 'stage3', 'stage4', 'preds']
    # stages = ['preds']
    
    nums = []
    for i, d in enumerate(domains):
        feat = np.array(np.load('D:/Med/DGDR/tsne/2w/feat_{}.npy'.format(d), allow_pickle=True))
        nums.append(feat.shape[0])
        # feat = np.sum(feat, axis=-1)
        # feat = np.sum(feat, axis=-1)
        feat = feat.reshape((feat.shape[0], -1))
        if i == 0:
            feats = feat
        else:
            feats = np.concatenate([feats, feat], axis=0)
            
    data_num, f_dim = feats.shape
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=10) # 30 100 10 50
    feature_vis = tsne.fit_transform(feats)
    font = {'family': 'Times New Roman','size': 15, 'weight': 'bold'}
    
    random_index = np.arange(nums[0] + nums[1] + nums[2] + nums[3] + nums[4] + nums[5])
    np.random.seed(123)
    random.shuffle(random_index)
    fig = plt.figure()
    ax = plt.axes()
    plt.subplot(111)

    for i in tqdm(range(feats.shape[0])):
        # bad
        if i < nums[0]:
            plt.scatter(feature_vis[i][0], feature_vis[i][1], color='CornflowerBlue', label='', alpha=1.0, s=5, marker='.')
        elif i >= nums[0] and i < nums[0] + nums[1]:
            plt.scatter(feature_vis[i][0], feature_vis[i][1], color='red', label='', alpha=1.0, s=5, marker='.')
        elif i >= nums[0] + nums[1] and i < nums[0] + nums[1] + nums[2]:
            plt.scatter(feature_vis[i][0], feature_vis[i][1], color='green', label='', alpha=1.0, s=5, marker='.')
        elif i >= nums[0] + nums[1] + nums[2] and i < nums[0] + nums[1] + nums[2] + nums[3]:
            plt.scatter(feature_vis[i][0], feature_vis[i][1], color='yellow', label='', alpha=1.0, s=5, marker='.')
        elif i >= nums[0] + nums[1] + nums[2] + nums[3] and i < nums[0] + nums[1] + nums[2] + nums[3] + nums[4]:
            plt.scatter(feature_vis[i][0], feature_vis[i][1], color='purple', label='', alpha=1.0, s=5, marker='.')
        elif i >= nums[0] + nums[1] + nums[2] + nums[3] + nums[4] and i < nums[0] + nums[1] + nums[2] + nums[3] + nums[4] + nums[5]:
            plt.scatter(feature_vis[i][0], feature_vis[i][1], color='pink', label='', alpha=1.0, s=5, marker='.')
        # good
        # if i < nums[0]:
        #     plt.scatter(feature_vis[random_index[i]][0], feature_vis[random_index[i]][1], color='CornflowerBlue', label='', alpha=1.0, s=5, marker='.')
        # elif i >= nums[0] and i < nums[0] + nums[1]:
        #     plt.scatter(feature_vis[random_index[i]][0], feature_vis[random_index[i]][1], color='red', label='', alpha=1.0, s=5, marker='.')
        # elif i >= nums[0] + nums[1] and i < nums[0] + nums[1] + nums[2]:
        #     plt.scatter(feature_vis[random_index[i]][0], feature_vis[random_index[i]][1], color='green', label='', alpha=1.0, s=5, marker='.')
        # elif i >= nums[0] + nums[1] + nums[2] and i < nums[0] + nums[1] + nums[2] + nums[3]:
        #     plt.scatter(feature_vis[random_index[i]][0], feature_vis[random_index[i]][1], color='yellow', label='', alpha=1.0, s=5, marker='.')
        # elif i >= nums[0] + nums[1] + nums[2] + nums[3] and i < nums[0] + nums[1] + nums[2] + nums[3] + nums[4]:
        #     plt.scatter(feature_vis[random_index[i]][0], feature_vis[random_index[i]][1], color='purple', label='', alpha=1.0, s=5, marker='.')
        # elif i >= nums[0] + nums[1] + nums[2] + nums[3] + nums[4] and i < nums[0] + nums[1] + nums[2] + nums[3] + nums[4] + nums[5]:
        #     plt.scatter(feature_vis[random_index[i]][0], feature_vis[random_index[i]][1], color='pink', label='', alpha=1.0, s=5, marker='.')
    
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # plt.legend(loc='lower right', prop=font)
    plt.savefig('vis/tsne/tsne_2w_10.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()