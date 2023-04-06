# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:01:05 2023

@author: gls
"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from palette import PALETTES


def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
    
    ax.add_artist(ellip)
    return ellip


class PCA_Analysis:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_scl = x
        self.pca = None
        
        
    def scale_data(self, with_mean = True, with_std = True):
        scl = StandardScaler(with_mean = with_mean, with_std = with_std)
        self.x_scl = scl.fit_transform(self.x)
        
        
    def perform_PCA(self, n_components = 2):
        pca = PCA(n_components = n_components)
        self.pca = pca
        return pca
    
    
    def plot_PCA(self, palette = 'lancet_lanonc'):
        y = self.y
        x_pca = self.pca.fit_transform(self.x_scl)
        colors = list(PALETTES[palette].values())
        plt.figure(dpi = 300)
        lbs = np.unique(y)
        for i, l in enumerate(lbs):
            k = np.where(y == l)[0]
            pts = x_pca[k,:]
            x1, x2 = pts.T
            plt.plot(x1, x2, '.', color=colors[i])
            plot_point_cov(pts, nstd=3, alpha=0.2, color=colors[i])
        plt.xlabel('PC1 ({} %)'.format(round(self.pca.explained_variance_ratio_[0] * 100, 2)))
        plt.ylabel('PC2 ({} %)'.format(round(self.pca.explained_variance_ratio_[1] * 100, 2)))
        
        
    
    
    