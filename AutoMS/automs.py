# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:04:55 2022

@author: DELL
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from AutoMS import hpic
from AutoMS import msdial
from AutoMS import library
from AutoMS import peakeval
from AutoMS import matching
from AutoMS import tandem
from AutoMS import deepmass
from AutoMS import analysis
from AutoMS import molnet
from AutoMS import enrichment


class AutoMSData:
    def __init__(self, ion_mode = 'positive'):
        """
        Initialize AutoMSData object.
        
        Parameters:
        - ion_mode (str): The ionization mode. Default is 'positive'.
        """
        self.data_path = None
        self.ion_mode = ion_mode
        self.peaks = None
        self.feature_table = None
        
        
    def load_files(self, data_path):
        """
        Load data files from the specified directory path.
        
        Parameters:
        - data_path (str): The path to the directory containing the data files.
        """
        self.data_path = data_path
        self.files = os.listdir(self.data_path)
    
        
    def find_features(self, min_intensity, mass_inv = 1, rt_inv = 30, min_snr = 3, max_items = 50000):
        """
        Find features in the loaded data files using HPIC algorithm.
        
        Parameters:
        - min_intensity (int): The minimum intensity threshold for peak detection.
        - mass_inv (int): The inverse of mass tolerance for clustering ions of the same metabolite. Default is 1.
        - rt_inv (int): The inverse of retention time tolerance for clustering ions of the same metabolite. Default is 30.
        - min_snr (int): The minimum signal-to-noise ratio threshold for peak detection. Default is 3.
        - max_items (int): The maximum number of ion traces to process. Default is 50000.
        """
        output = {}
        files = self.files
        for i, f in enumerate(files):
            print('processing {}, {}/{} files, set maximum {} ion traces'.format(f, 1+i, len(files), max_items))
            peaks, pics = hpic.hpic(os.path.join(self.data_path, f), 
                                    min_intensity = min_intensity, 
                                    min_snr = min_snr, 
                                    mass_inv = mass_inv, 
                                    rt_inv = rt_inv,
                                    max_items = max_items)
            output[f] = {'peaks': peaks, 'pics': pics}
        self.peaks = output
        

    def evaluate_features(self):
        """
        Evaluate the extracted features using peak evaluation.

        Raises:
        - ValueError: If no peaks are found. Please find peaks first before evaluating.
        """
        if self.peaks is None:
            raise ValueError('Please find peak first')
        for f, vals in self.peaks.items():
            peak = vals['peaks']
            pic = vals['pics']
            score = peakeval.evaluate_peaks(peak, pic)
            self.peaks[f]['peaks']['score'] = score


    def match_features(self, method = 'simple', mz_tol = 0.01, rt_tol = 20, min_frac = 0.5):
        """
        Match and filter extracted features using feature matching.

        Parameters:
        - method (str): The feature matching method to use. Default is 'simple'.
        - mz_tol (float): The mass-to-charge ratio tolerance for feature matching. Default is 0.01.
        - rt_tol (float): The retention time tolerance for feature matching. Default is 20.
        - min_frac (float): The minimum fraction of samples that should have a feature for it to be considered. Default is 0.5.

        Raises:
        - ValueError: If no peaks are found. Please find peaks first before matching.
        - IOError: If an invalid method is provided.
        """
        if self.peaks is None:
            raise ValueError('Please find peak first')
        linker = matching.FeatureMatching(self.peaks)
        if method == 'simple':
            linker.simple_matching(mz_tol = mz_tol, rt_tol = rt_tol)
        else:
            raise IOError('Invalid Method')
        self.feature_table = linker.feature_filter(min_frac = min_frac)
        self.feature_table['Ionmode'] = self.ion_mode


    def load_features_msdial(self, msdial_path):
        """
        Load feature extraction results from MS-DIAL into AutoMS.

        Parameters:
        - msdial_path (str): The path to the MS-DIAL feature extraction result file.
        """
        data_path = self.data_path
        self.peaks = {f: {} for f in os.listdir(data_path)}
        self.feature_table = msdial.load_msdial_result(data_path, msdial_path)
        self.feature_table['Ionmode'] = self.ion_mode

    
    def match_features_with_ms2(self, mz_tol = 0.01, rt_tol = 15):
        """
        Match features with corresponding MS/MS spectra.

        Parameters:
            - mz_tol (float): The m/z tolerance for matching features with spectra. Default is 0.01.
            - rt_tol (float): The retention time tolerance for matching features with spectra. Default is 15.
        """
        files = [os.path.join(self.data_path, f) for f in list(self.peaks.keys())]
        spectrums = tandem.load_tandem_ms(files)
        spectrums = tandem.cluster_tandem_ms(spectrums, mz_tol = mz_tol, rt_tol = rt_tol)
        self.feature_table = tandem.feature_spectrum_matching(self.feature_table, spectrums, mz_tol = mz_tol, rt_tol = rt_tol)
    
    
    def search_library(self, lib_path, method = 'entropy', ms1_da = 0.01, ms2_da = 0.05, threshold = 0.5):
        """
        Search a library for metabolite annotation based on the feature table.
    
        Parameters:
            - lib_path (str): The path to the library file.
            - method (str): The method for library search. Default is 'entropy'.
            - ms1_da (float): The m/z tolerance for matching MS1 masses. Default is 0.01.
            - ms2_da (float): The m/z tolerance for matching MS2 masses. Default is 0.05.
            - threshold (float): The annotation confidence threshold. Default is 0.5.
        """
        feature_table = self.feature_table
        value_columns = list(self.peaks.keys())
        lib = library.SpecLib(lib_path)
        self.feature_table_annotated = lib.search(feature_table = feature_table, method = method, ms1_da = ms1_da, ms2_da = ms2_da, threshold = threshold)
        self.feature_table_annotated = self.refine_annotated_table(value_columns = value_columns)


    def refine_annotated_table(self, value_columns):
        """
        Refine the annotated feature table by selecting representative entries for each annotated compound.
    
        Parameters:
            - value_columns (list): The list of columns containing values to consider for selecting representative entries.
            
        Returns:
            - feature_table_annotated (DataFrame): The refined feature table with representative entries for each compound.
        """
        feature_table = self.feature_table
        print('refine feature table with annotation')
        keep = []
        uni_comp = list(set(feature_table['Annotated Name']))
        for comp in tqdm(uni_comp):
            if comp is None:
                continue
            wh = np.where(feature_table['Annotated Name'] == comp)[0]
            if len(wh) == 1:
                keep.append(wh[0])
            else:
                mean_vals = np.mean(feature_table.loc[wh, value_columns].values, axis = 1)
                keep.append(wh[np.argmax(mean_vals)])
        keep = np.sort(keep)
        feature_table_annotated = feature_table.loc[keep,:]
        return feature_table_annotated

        
    def load_external_annotation(self, annotation_file, mz_tol = 0.01, rt_tol = 10):
        feature_table = self.feature_table
        annotation_table = pd.read_csv(annotation_file)
        annotation_table.loc[:,'MZ'] = annotation_table.loc[:,'MZ'].astype(float)
        annotation_table.loc[:,'RT'] = annotation_table.loc[:,'RT'].astype(float)
        for i in feature_table.index:
            k1 = np.abs(float(feature_table.loc[i, 'MZ']) - annotation_table.loc[:,'MZ']) < mz_tol
            k2 = np.abs(float(feature_table.loc[i, 'RT']) - annotation_table.loc[:,'RT']) < rt_tol
            kk = np.where(np.logical_and(k1, k2))[0]
            if len(kk) == 0:
                continue
            elif len(kk) > 1:
                k = kk[np.argmin(np.abs(feature_table.loc[i, 'MZ'] - annotation_table.loc[kk,'MZ']))]
            else:
                k = kk[0]
            feature_table.loc[i, 'Annotated Name'] = annotation_table.loc[k,'Name']
            feature_table.loc[i, 'InChIKey'] = annotation_table.loc[k,'InChIKey']
            feature_table.loc[i, 'SMILES'] = annotation_table.loc[k,'SMILES']
            feature_table.loc[i, 'Matching Score'] = 'external annotation'
        self.feature_table = feature_table
        

    def export_ms2_to_mgf(self, save_path):
        deepmass.export_to_mgf(self.feature_table, save_path)
    
    
    def load_deepmass_annotation(self, deepmass_dir):
        value_columns = list(self.peaks.keys())
        self.feature_table = deepmass.link_to_deepmass(self.feature_table, deepmass_dir)
        self.feature_table_annotated = self.refine_annotated_table(self.feature_table, value_columns)
        
    def save_project(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    
    def load_project(self, save_path):
        with open(save_path, 'rb') as f:
            obj_dict = pickle.load(f)
        self.__dict__.update(obj_dict)
        

    def export_features(self):
        return AutoMSFeature(self.feature_table, self.files)




class AutoMSFeature:
    def __init__(self, feature_table = None, sample_list = None):
        self.feature_table = feature_table
        self.files = sample_list
        self.feature_table_annotated = None
        self.biomarker_table = None
        
    
    def update_sample_name(self, namelist):
        if len(namelist) != len(self.files):
            raise IOError('the length of input name list is not equal to the length of original name list')
        colnames = list(self.feature_table)
        i = colnames.index(self.files[0])
        j = colnames.index(self.files[-1]) + 1
        colnames[i:j] = namelist
        self.feature_table.columns = colnames
        self.files = namelist
        print('after update sample name, please re-do <refine_annotated_table> and <select_biomarker> to update related tables')
    
    
    def append_feature_table(self, feature_object):
        if type(feature_object) != AutoMSFeature:
            raise IOError('input is not a AutoMSFeature object')
        if self.feature_table is None:
            self.files = feature_object.files
            self.feature_table = feature_object.feature_table
        else:
            if not (np.array(self.files) ==  np.array(feature_object.files)).all():
                raise IOError('input feature object has different samples from the original')
            self.feature_table = pd.concat([self.feature_table, feature_object.feature_table], ignore_index = True)
        print('after append another feature table, please re-do <refine_annotated_table> and <select_biomarker> to update related tables')
    

    def preprocessing(self, impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5, qc_samples = None, group_info = None, **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        files = self.files
        intensities = self.feature_table[files]
        count_nan = np.sum(~np.isnan(intensities), axis = 1)
        wh = np.where(count_nan / intensities.shape[1] >= min_frac)[0]
        self.feature_table = self.feature_table.loc[wh,:]
        self.feature_table = self.feature_table.reset_index(drop = True) 
        x = self.feature_table.loc[:,files]
        preprocessor = analysis.Preprocessing(x)
        x_prep = preprocessor.one_step(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5, qc_samples = qc_samples, group_info = group_info, **args)
        self.feature_table.loc[:,files] = x_prep

    
    def refine_annotated_table(self):
        feature_table = self.feature_table
        value_columns = self.files
        print('refine feature table with annotation')
        keep = []
        try:
            uni_comp = list(set(feature_table['Annotated Name']))
        except:
            raise ValueError('There is no annotation information')
        for comp in tqdm(uni_comp):
            if comp is None:
                continue
            wh = np.where(feature_table['Annotated Name'] == comp)[0]
            if len(wh) == 1:
                keep.append(wh[0])
            else:
                mean_vals = np.mean(feature_table.loc[wh, value_columns].values, axis = 1)
                keep.append(wh[np.argmax(mean_vals)])
        keep = np.sort(keep)
        self.feature_table_annotated = feature_table.loc[keep,:]

    
    def perform_dimensional_reduction(self, group_info = None, method = 'PCA', annotated_only = True, **args):
        if annotated_only:
            feature_table = self.feature_table_annotated
        else:
            feature_table = self.feature_table
        if feature_table is None:
            raise ValueError('Please match peak or load deepmass result, first')
            
        files = self.files
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            x = feature_table.loc[:,files].T
            y = np.repeat('Samples', len(files))
            
        DRAnal = analysis.Dimensional_Reduction(x, y)
        DRAnal.scale_data()
        if method == 'PCA':
            DRAnal.perform_PCA(**args)
        elif method == 'tSNE':
            DRAnal.perform_tSNE(**args)
        elif method == 'uMAP':
            DRAnal.perform_uMAP(**args)
        else:
            raise IOError('Invalid Method')
        DRAnal.plot_2D()
        
    
    def perform_PLSDA(self, group_info = None, n_components=2, n_permutations = 1000, annotated_only = True, loo_test = True, permutation_test = True):
        if annotated_only:
            feature_table = self.feature_table_annotated
        else:
            feature_table = self.feature_table
        if feature_table is None:
            raise ValueError('Please match peak or load deepmass result, first')
            
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        plsda = analysis.PLSDA(x, y, n_components = n_components)
        plsda.scale_data()
        plsda.perform_PLSDA()
        plsda.plot_2D()
        if loo_test:
            plsda.leave_one_out_test()
        if permutation_test:
            plsda.perform_permutation_test(n_permutations = n_permutations)
        
        if annotated_only:
            self.feature_table_annotated['PLS_VIP'] = plsda.get_VIP()   
        else:
            self.feature_table['PLS_VIP'] = plsda.get_VIP()   
    
    
    def perform_RandomForest(self, group_info = None, annotated_only = True, loo_test = True, **args):
        if annotated_only:
            feature_table = self.feature_table_annotated
        else:
            feature_table = self.feature_table
        if feature_table is None:
            raise ValueError('Please match peak or load deepmass result, first')
            
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        rf = analysis.RandomForest(x, y)
        rf.scale_data()
        rf.perform_RF(**args)

        if annotated_only:
            self.feature_table_annotated['RF_VIP'] = rf.get_VIP() 
        else:
            self.feature_table['RF_VIP'] = rf.get_VIP() 
        
        if loo_test:
            rf.leave_one_out_test()
        else:
            rf.out_of_bag_score()


    def perform_GradientBoost(self, model = 'XGBoost', group_info = None, annotated_only = True, loo_test = True, **args):
        if annotated_only:
            feature_table = self.feature_table_annotated
        else:
            feature_table = self.feature_table
        if feature_table is None:
            raise ValueError('Please match peak or load deepmass result, first')
            
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        xgb = analysis.GradientBoost(x, y)
        xgb.scale_data()
        if model == 'XGBoost':
            xgb.perform_XGBoost(**args)
        elif model == 'LightGBM':
            xgb.perform_LightGBM(**args)
        else:
            raise IOError('invalid model')

        if annotated_only:
            self.feature_table_annotated['GradientBoost_VIP'] = xgb.get_VIP() 
        else:
            self.feature_table['GradientBoost_VIP'] = xgb.get_VIP() 
        
        if loo_test:
            xgb.leave_one_out_test()
    
    
    def perform_T_Test(self, group_info = None, annotated_only = True, **args):
        if annotated_only:
            feature_table = self.feature_table_annotated
        else:
            feature_table = self.feature_table
        if feature_table is None:
            raise ValueError('Please match peak or load deepmass result, first')
            
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = feature_table.loc[:,files_keep]
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        t_test = analysis.T_Test(x, y)
        t_test.perform_t_test()
        t_test.calc_fold_change()
        t_test.perform_multi_test_correlation(**args)
        t_test.plot_volcano()
        
        if annotated_only:
            self.feature_table_annotated['T_Test_P_{}'.format('_'.join(group_info.keys()))] = t_test.p_values
        else:
            self.feature_table['T_Test_P_{}'.format('_'.join(group_info.keys()))] = t_test.p_values
    
    
    def select_biomarker(self, criterion = {'PLS_VIP': ['>', 1.5]}, combination = 'union', annotated_only = True):
        if annotated_only:
            feature_table = self.feature_table_annotated
        else:
            feature_table = self.feature_table
        if feature_table is None:
            raise ValueError('Please match peak or load deepmass result, first')
        
        if combination == 'union':
            selected = np.repeat(False, len(feature_table))
        else:
            selected = np.repeat(True, len(feature_table))
        for i, cir in criterion.items():
            if i not in feature_table.columns:
                raise ValueError('{} not in columns of feature table, please check if {} is calculated'.format(i,i))
            vals = feature_table.loc[:, i].values
            thres = cir[1]
            if cir[0] == '>':
                res_i = vals > thres
            elif cir[0] == '>=':
                res_i = vals >= thres
            elif cir[0] == '<=':
                res_i = vals <= thres
            elif cir[0] == '<':
                res_i = vals < thres
            else:
                raise ValueError('{} is not a compare function'.format(cir[0]))
            if combination == 'union':
                selected = np.logical_or(selected, res_i)
            else:
                selected = np.logical_and(selected, res_i)
        self.biomarker_table = feature_table.loc[selected, :]
        
    
    def perform_heatmap(self, biomarker_only = True, group_info = None, hide_xticks = False, hide_ytick = False):
        # to do: 
            # highlight biomarker
        if biomarker_only:
            biomarker_table = self.biomarker_table
        else:
            biomarker_table = self.feature_table_annotated
        if len(biomarker_table) >= 200:
            raise IOError('Too many features selected')
        files = self.files
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = biomarker_table.loc[:,files_keep]
        else:
            x = biomarker_table.loc[:,files]
        x_mean = np.mean(x, axis = 1)
        x_ = np.log2(x.div(x_mean, axis=0))
        plt.figure(dpi = 300)
        if 'Annotated Name' in list(biomarker_table.columns):
            yticklabels = [s[:30] for s in list(biomarker_table['Annotated Name'])]
        else:
            yticklabels = True
        if hide_ytick:
            yticklabels = False
        if hide_xticks:
            xticklabels = False
            sns.clustermap(x_, cmap="bwr", figsize = (8, len(x_) / 5), xticklabels = xticklabels, yticklabels = yticklabels, vmin=-np.max(np.abs(x_.values)), vmax=np.max(np.abs(x_.values)))
        else:
            sns.clustermap(x_, cmap="bwr", figsize = (8, len(x_) / 5), yticklabels = yticklabels, vmin=-np.max(np.abs(x_.values)), vmax=np.max(np.abs(x_.values)))

        
    def perform_molecular_network(self, threshold = 0.5, target_compound = None, group_info = None):
        feature_table = self.feature_table_annotated
        if feature_table is None:
            raise ValueError('the selected table is None')
        net = molnet.MolNet(feature_table, group_info)
        net.compute_similarity_matrix()
        net.create_network(threshold = threshold)
        net.plot_global_network()
        if target_compound is not None:
            net.get_subgraph(target_compound = target_compound)
            net.plot_selected_subgraph()
    
    
    def perform_enrichment_analysis(self, organism="hsa", pvalue_cutoff = 0.05, adj_method = "fdr_bh"):
        biomarker_table = self.biomarker_table
        inchikey_list = biomarker_table['InChIKey']
        enrichment_analysis = enrichment.EnrichmentAnalysis(inchikey_list, organism = organism)
        enrichment_analysis.run_analysis(pvalue_cutoff=0.05)
        results = enrichment_analysis.results
        enrichment.plot_enrichment_analysis_results(results, adj_method = adj_method)
        
    
    def save_project(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    
    def load_project(self, save_path):
        with open(save_path, 'rb') as f:
            obj_dict = pickle.load(f)
        self.__dict__.update(obj_dict)



if __name__ == '__main__':
    
    pass