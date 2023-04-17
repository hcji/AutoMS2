# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:01:05 2023

@author: gls
"""


import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from tqdm import tqdm
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from seaborn import heatmap
from adjustText import adjust_text

from AutoMS import imputer
from AutoMS.palette import PALETTES



def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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


class Preprocessing:
    def __init__(self, x):
        self.x = x
        self.x_out = x
    
    
    def normalize(self, method = 'median'):
        sample_median = np.median(x, axis = 0)
        overall_median = np.median(sample_median)
        normalized_coeff = sample_median / overall_median
        x_norm = x / normalized_coeff
        self.x_out = x_norm
    
    
    def calc_RSD(self, qc_samples = None):
        if qc_samples is not None:
            x_qc = self.x.loc[:, qc_samples]
        else:
            x_qc = self.x
        x_rsd = np.nanstd(x_qc, axis = 1) / np.nanmean(x_qc, axis = 1)
        
        plt.figure(dpi = 300)
        plt.hist(x_rsd * 100, bins = int(len(x_rsd) / 200), color = '#4DBBD5')
        plt.axvline(30, color = '#E64B35', linestyle='--')
        plt.xlabel('Relative Standard Deviation (%)', fontsize = 12)
        plt.ylabel('Frequency', fontsize = 12)
        return x_rsd
    
    
    def impute_missing_features(self, impute_method = 'KNN', **args):
        imp = imputer.Imputer(self.x, None)
        if impute_method == 'Low value':
            x_imp = imp.fill_with_low_value()
        elif impute_method == 'Mean':
            x_imp = imp.fill_with_mean_value()
        elif impute_method == 'Median':
            x_imp = imp.fill_with_median_value()
        elif impute_method == 'KNN':
            x_imp = imp.fill_with_knn_imputer(**args)
        elif impute_method == 'Iterative RF':
            x_imp = imp.fill_with_iterative_RF(**args)
        elif impute_method == 'Iterative BR':
            x_imp = imp.fill_with_iterative_BR(**args)
        elif impute_method == 'Iterative SVR':
            x_imp = imp.fill_with_iterative_SVR(**args)
        else:
            raise ValueError(f"Invalid imputation method: {impute_method}")
        self.x_out = pd.DataFrame(x_imp[0], columns = self.x.columns)
    
    
    def filter_outlier(self, group_info = None, outlier_threshold = 3):
        x = self.x
        if group_info is None:
            group_info = {'all': x.columns}
        for key, cols in group_info.items():
            x_sub = x.loc[:,cols]
            x_medians = np.median(x_sub, axis = 1)
            x_stds = np.std(x_sub, axis = 1)
            x_outlier_indices = np.abs(x_sub - x_medians[:, np.newaxis]) > outlier_threshold * x_stds[:, np.newaxis]
            for col in x_sub:
                sample = x_sub[col]
                outliers = x_outlier_indices[col]
                sample[outliers] = np.nan
                x_sub[col] = sample
            x.loc[:,cols] = x_sub
        self.x_out = x
    
    
    def filter_RSD(self, qc_samples = None, rsd_threshold = 0.3):
        x = self.x
        x_rsd = self.calc_RSD(qc_samples = qc_samples)
        keep = np.where(x_rsd <= rsd_threshold)[0]
        x = x.loc[keep,:]
        self.x_out = x.reset_index(drop = True)


    def plot_correlation(self, qc_samples = None):
        if qc_samples is not None:
            x_qc = self.x_out.loc[:, qc_samples]
        else:
            x_qc = self.x_out
        scl = StandardScaler()
        x_scl = pd.DataFrame(scl.fit_transform(x_qc), columns = x_qc.columns)
        corr_mat = x_scl.corr()
        plt.figure(dpi = 300)
        heatmap(corr_mat, cmap="RdBu_r", annot=True, vmin=0.5, vmax=1)


    def one_step(self, impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, qc_samples = None, group_info = None, **args):
        print('impute missing values #1')
        self.impute_missing_features(impute_method = impute_method, **args)
        print('filter invalid features with RSDs (with QC samples)')
        self.filter_RSD(qc_samples = qc_samples, rsd_threshold = 0.3)
        print('filter outliers')
        self.filter_outlier(group_info = group_info, outlier_threshold = 3)
        print('impute missing values #2')
        self.impute_missing_features(impute_method = impute_method, **args)
        print('calculate correlation (of QC samples)')
        self.plot_correlation(qc_samples = qc_samples)
        return self.x_out


class T_Test:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.lbs = np.unique(y)
        self.p_values = np.repeat(np.nan, len(x))
        self.log2FC = np.repeat(np.nan, len(x))
        if len(self.lbs) != 2:
            raise IOError('Invalid input of y')
        
        
    def perform_t_test(self):
        group1 = self.x[:, self.y == self.lbs[0]]
        group2 = self.x[:, self.y == self.lbs[1]]
        print('perform t test...')
        for i in tqdm(range(group1.shape[0])):
            _, p = ttest_ind(group1[i], group2[i])
            self.p_values[i] = p
    
    
    def perform_multi_test_correlation(self, alpha=0.05, method='fdr_bh'):
        reject, self.p_values, _, _ = multipletests(self.p_values, alpha=alpha, method=method)
        
        
    def calc_fold_change(self):
        group1 = self.x[:, self.y == self.lbs[0]]
        group2 = self.x[:, self.y == self.lbs[1]]
        print('calculate fold change...')
        for i in tqdm(range(group1.shape[0])):
            m1 = np.mean(group1[i])
            m2 = np.mean(group2[i])
            self.log2FC[i] = np.log2(m1 / m2)
    
    
    def plot_volcano(self, feature_name = None, fc_threshold = 2.0, pval_threshold = 0.05, topN = 20):
        up_regulated = (self.log2FC > fc_threshold) & (self.p_values < pval_threshold)
        down_regulated = (self.log2FC < -fc_threshold) & (self.p_values < pval_threshold)
        not_significant = (self.p_values >= pval_threshold)
        mixed = ~(up_regulated | down_regulated | not_significant)
        colors = {'Up': '#E64B35', 'Down': '#4DBBD5', 'NS': 'grey', 'Mixed': '#00A087'}
        labels = {'Up': 'Up-regulated', 'Down': 'Down-regulated', 'NS': 'Not significant', 'Mixed': 'No diff'}
        
        plt.figure(dpi = 300)
        plt.scatter(self.log2FC[up_regulated], -np.log10(self.p_values[up_regulated]), color=colors['Up'], label=labels['Up'])
        plt.scatter(self.log2FC[down_regulated], -np.log10(self.p_values[down_regulated]), color=colors['Down'], label=labels['Down'])
        plt.scatter(self.log2FC[not_significant], -np.log10(self.p_values[not_significant]), color=colors['NS'], label=labels['NS'])
        plt.scatter(self.log2FC[mixed], -np.log10(self.p_values[mixed]), color=colors['Mixed'], label=labels['Mixed'])
        plt.axhline(y = -np.log10(pval_threshold), color='black', linestyle='--')
        plt.axvline(x = fc_threshold, color='black', linestyle='--')
        plt.axvline(x = -fc_threshold, color='black', linestyle='--')
        plt.xlabel('log2(Fold change)')
        plt.ylabel('-log10(p value)')
        plt.legend()
        
        if feature_name is None:
            return
        texts = []
        topN_threshold = np.sort(self.p_values)[min(topN, len(self.p_values))]
        for i in range(len(mixed)):
            if (up_regulated[i] or down_regulated[i]) and self.p_values[i] <= topN_threshold:
                x, y, s = self.log2FC[i], -np.log10(self.p_values[i]), feature_name[i]
                texts.append(plt.text(x, y, s, fontsize = 8))
        adjust_text(texts, force_points=0.2, force_text=0.2,
                    expand_points=(1, 1), expand_text=(1, 1),
                    arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
        
            
        


class Dimensional_Reduction:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_scl = x
        self.model = None
        
        
    def scale_data(self, with_mean = True, with_std = True):
        scl = StandardScaler(with_mean = with_mean, with_std = with_std)
        self.x_scl = scl.fit_transform(self.x)
        
        
    def perform_PCA(self, n_components = 2):
        pca = PCA(n_components = n_components).fit(self.x_scl)
        self.model = pca

    
    def perform_tSNE(self, n_components = 2):
        tsne = TSNE(n_components=n_components).fit(self.x_scl)
        self.model = tsne
        
    
    def perform_uMAP(self, n_neighbors=5, min_dist=0.3):
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit(self.x_scl)
        self.model = umap
    
    
    def plot_2D(self, palette = 'lancet_lanonc'):
        y = np.array(self.y)
        x_map = self.model.fit_transform(self.x_scl)
        colors = list(PALETTES[palette].values())
        plt.figure(dpi = 300)
        lbs = np.unique(y)
        for i, l in enumerate(lbs):
            k = np.where(y == l)[0]
            pts = x_map[k,:]
            x1, x2 = pts.T
            plt.plot(x1, x2, '.', color=colors[i], label = l)
            plot_point_cov(pts, nstd=3, alpha=0.2, color=colors[i])
        
        if type(self.model) == sklearn.decomposition._pca.PCA:
            plt.xlabel('PC1 ({} %)'.format(round(self.model.explained_variance_ratio_[0] * 100, 2)))
            plt.ylabel('PC2 ({} %)'.format(round(self.model.explained_variance_ratio_[1] * 100, 2)))
        elif type(self.model) == sklearn.manifold._t_sne.TSNE:
            plt.xlabel('tSNE 1')
            plt.ylabel('tSNE 2')
        else:
            plt.xlabel('uMAP 1')
            plt.ylabel('uMAP 2')
        plt.legend()
        

class PLSDA:
    def __init__(self, x, y, n_components=2):
        self.x = x
        self.y = y
        self.n_components = n_components
        self.lab_enc = LabelEncoder()
        self.y_label = self.lab_enc.fit_transform(y)
        self.x_scl = x
        self.pls = None
        self.lda = None
        
    
    def scale_data(self, with_mean = True, with_std = True):
        scl = StandardScaler(with_mean = with_mean, with_std = with_std)
        self.x_scl = scl.fit_transform(self.x)    
        

    def perform_PLSDA(self):
        self.pls = PLSRegression(n_components=self.n_components)
        x_pls = self.pls.fit_transform(self.x, self.y_label)[0]
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(x_pls, self.y_label)
        
    
    def get_VIP(self):
        t = self.pls.x_scores_
        w = self.pls.x_weights_
        q = self.pls.y_loadings_
        m, p = self.x_scl.shape
        _, h = t.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return vips


    def plot_2D(self, palette = 'lancet_lanonc'):
        y = np.array(self.y)
        x_map = self.pls.transform(self.x_scl)
        colors = list(PALETTES[palette].values())
        plt.figure(dpi = 300)
        lbs = np.unique(y)
        for i, l in enumerate(lbs):
            k = np.where(y == l)[0]
            pts = x_map[k,:]
            x1, x2 = pts.T
            plt.plot(x1, x2, '.', color=colors[i], label = l)
            plot_point_cov(pts, nstd=3, alpha=0.2, color=colors[i])
        plt.legend()
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        
        
    def leave_one_out_test(self, y_new = None, plot = True):
        X = np.array(self.x)
        if y_new is None:
            y = self.y_label
        else:
            y = y_new
        class_labels = self.lab_enc.classes_
        
        pls = PLSRegression(n_components = self.n_components)
        lda = LinearDiscriminantAnalysis()
        n_samples = X.shape[0]
        y_preds = []
        for i in range(n_samples):
            test_indices = [i]
            train_indices = list(set(range(n_samples)) - set(test_indices))

            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]

            X_train_pls = pls.fit_transform(X_train, y_train)[0]
            X_test_pls = pls.transform(X_test)

            lda.fit(X_train_pls, y_train)
            y_preds.append(lda.predict(X_test_pls)[0])
        
        if not plot:
            return accuracy_score(y, y_preds)
        
        print('Classification matrics of leave-one-out test')
        print(classification_report(y, y_preds, target_names=class_labels))
        
        confusion = confusion_matrix(y, y_preds)
        norm_confusion_matrix = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(dpi = 300)
        im = ax.imshow(norm_confusion_matrix, cmap='Blues')
        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        thresh = norm_confusion_matrix.max() / 2.
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                ax.text(j, i, format(norm_confusion_matrix[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if norm_confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()     
        plt.show()


    def perform_permutation_test(self, n_permutations = 1000):
        accuracies = []
        true_accuracy = self.leave_one_out_test(plot = False)
        print("Perform permutation test")
        for i in tqdm(range(n_permutations)):
            y_permuted = np.random.permutation(self.y_label)
            acc = self.leave_one_out_test(y_new = y_permuted, plot = False)
            accuracies.append(acc)
        p_value = np.mean(accuracies >= true_accuracy)
        print("\nTrue accuracy:", true_accuracy)
        print("p-value:", p_value)
        
        plt.figure(dpi = 300)
        plt.hist(accuracies, bins = min(int(n_permutations / 25), self.x.shape[0]), color = '#4DBBD5')
        plt.axvline(true_accuracy, color = '#E64B35', linestyle='--')
        plt.xlabel('Accuracy of LOO', fontsize = 12)
        plt.ylabel('Frequency', fontsize = 12)



class RandomForest:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_scl = x
        self.model = None
        
    
    def scale_data(self, with_mean = True, with_std = True):
        scl = StandardScaler(with_mean = with_mean, with_std = with_std)
        self.x_scl = scl.fit_transform(self.x)      


    def perform_RF(self, **args):
        self.model = RandomForestClassifier(oob_score = True, **args)
        self.model.fit(self.x, self.y)
    
    
    def out_of_bag_score(self):
        y = self.y
        class_labels = self.model.classes_
        oob_decision = self.model.oob_decision_function_
        y_preds = self.model.classes_[np.argmax(oob_decision, axis = 1)]
        
        print('Classification matrics of out-of-bag')
        print(classification_report(y, y_preds, target_names=class_labels))
        
        confusion = confusion_matrix(y, y_preds)
        norm_confusion_matrix = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(dpi = 300)
        im = ax.imshow(norm_confusion_matrix, cmap='Blues')
        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        thresh = norm_confusion_matrix.max() / 2.
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                ax.text(j, i, format(norm_confusion_matrix[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if norm_confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()     
        plt.show()


    def get_VIP(self):
        vips = self.model.feature_importances_
        return vips


    
if __name__ == '__main__':
    
    from sklearn import datasets
    
    # example data
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target_names[iris.target]
    
    # Dimensional Reduction
    dra = Dimensional_Reduction(x, y)
    dra.scale_data()
    dra.perform_PCA()
    dra.plot_2D()
    dra.perform_tSNE()
    dra.plot_2D()
    dra.perform_uMAP()
    dra.plot_2D()
    
    # PLS-DA
    plsda = PLSDA(x, y)
    plsda.scale_data()
    plsda.perform_PLSDA()
    plsda.plot_2D()
    plsda.leave_one_out_test()
    plsda.perform_permutation_test()
    
    
    