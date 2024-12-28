import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from tqdm import tqdm
import cv2
from PIL import Image
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.pyplot import MultipleLocator
import random



def getids(start, end, n):
    return random.sample([i for i in range(start, end+1)], n)



if __name__ == '__main__':
    
    # status = ['good', 'bad']
    # domains = ['0', '1', '2', '3']
    domains = ['APTOS', 'DEEPDR', 'FGADR', 'IDRID', 'MESSIDOR', 'RLDR']
    # stages = ['stage1', 'stage2', 'stage3', 'stage4']
    
    # for s in stages:
    for d in domains:
        feat_x = np.array(np.load(os.path.join('D:/Med/DGDR/stat', 'mu_{}.npy'.format(d)), allow_pickle=True))
        feat_y = np.array(np.load(os.path.join('D:/Med/DGDR/stat', 'sigma_{}.npy'.format(d)), allow_pickle=True))
        # feat = np.sum(feat_x, axis=-1)
        # feat = np.sum(feat, axis=-1)
        feats_x = feat_x.reshape((feat_x.shape[0], -1))
        feats_y = feat_y.reshape((feat_y.shape[0], -1))
        feats_x = np.mean(feats_x, axis=-1)
        feats_y = np.mean(feats_y, axis=-1)
        
        ## norm
        feats_x = (feats_x - feats_x.min()) / (feats_x.max() - feats_x.min())
        feats_y = (feats_y - feats_y.min()) / (feats_y.max() - feats_y.min())
            
        data_num = feats_x.shape[0]
        # x = np.arange(c_dim)
        x = feats_x * 0.7 + np.random.normal(0, 0.05, feats_x.shape)
        y = feats_y * 0.7 + np.random.normal(0, 0.05, feats_y.shape)

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        
        fig = plt.figure(figsize=(12, 9))
        ax = plt.axes()
        
        for ind in range(data_num):
            plt.scatter(x, y, label='sample', color='cornflowerblue', marker='o', s=12)
        # ax.set_xticklabels(x)
        # plt.xticks(fontproperties='Times New Roman', size=28, weight='bold', rotation=90)
        plt.xticks(fontproperties='Times New Roman', size=28, weight='bold')
        plt.yticks(fontproperties='Times New Roman', size=28, weight='bold')
        # plt.ylim(0, 1)
        # y_major_locator = MultipleLocator(0.5)
        # ax = plt.gca()
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.ylabel('mu', fontproperties='Times New Roman', size=30, weight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        plt.savefig('vis/GOOD_stat/{}_stat.png'.format(d), dpi=600, bbox_inches='tight', pad_inches=0.1)
        print('{} Done'.format(d))