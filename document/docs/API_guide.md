# API Guide

This is an overview over all classes available in AutoMS.

## AutoMSData
The AutoMSData class represents an object for handling mass spectrometry data.

**Methods:**

- **__init__(self, ion_mode='positive')**: Initialize the AutoMSData object with an optional ion_mode parameter specifying the ionization mode.
- **load_files(self, data_path)**: Load data files from the specified directory path.
- **find_features(self, min_intensity, mass_inv=1, rt_inv=30, min_snr=3, max_items=50000)**: Find features in the loaded data files using the HPIC algorithm. 
- **evaluate_features(self)**: Evaluate the extracted features using peak evaluation.
- **match_features(self, method = 'simple', mz_tol = 0.01, rt_tol = 20, min_frac = 0.5)**: Match and filter extracted features using feature matching.
- **load_features_msdial(self, msdial_path)**: Load feature extraction results from MS-DIAL into AutoMS.
- **match_features_with_ms2(self, mz_tol = 0.01, rt_tol = 15)**: Match features with corresponding MS/MS spectra.
- **search_library(self, lib_path, method = 'entropy', ms1_da = 0.01, ms2_da = 0.05, threshold = 0.5)**: Search a library for metabolite annotation based on the feature table.
- **refine_annotated_table(self, value_columns)**: Refine the annotated feature table by selecting representative entries for each annotated compound.
- **load_external_annotation(self, annotation_file, mz_tol = 0.01, rt_tol = 10)**: Load external annotation information from a file and match it with the feature table.
- **export_ms2_to_mgf(self, save_path)**: Export feature MS2 spectra to an MGF file.
- **load_deepmass_annotation(self, deepmass_dir)**: Load annotation information from DeepMass results and link it with the feature table.
- **save_project(self, save_path)**: Save the current project to a file.
- **load_project(self, save_path)**: Load a project from a file.


**Attributes:**

- **data_path**: str, The path of the dataset of the experiment.
- **ion_mode**: str, The ionization mode of the experiment.
- **peaks**: dict, Extracted peaks via HPIC, which will be *None* if features are load from exterinal software.
- **feature_table**: DataFrame, Extracted features with all obtained information from the executed methods.


## AutoMSFeature

**Methods:**

- **__init__(self, feature_table=None, sample_list=None)**: Initialize AutoMSFeature object.  
- **update_sample_name(self, namelist)**: Update the sample names in the feature table.  
- **append_feature_table(self, feature_object)**: Append another feature table to the existing feature table.  
- **preprocessing(self, impute_method='KNN', outlier_threshold=3, rsd_threshold=0.3, min_frac=0.5, qc_samples=None, group_info=None)**: Preprocess the feature table by imputing missing values, removing outliers, and scaling the data.
- **refine_annotated_table(self)**: Refine the feature table with annotation by selecting representative entries for each compound.
- **perform_dimensional_reduction(self, group_info=None, method='PCA', annotated_only=True)**: Perform dimensional reduction on the feature table for visualization.
- **perform_PLSDA(self, group_info=None, n_components=2, n_permutations=1000, annotated_only=True, loo_test=True, permutation_test=True)**: Perform Partial Least Squares Discriminant Analysis (PLSDA) on the feature table.
- **perform_RandomForest(self, group_info=None, annotated_only=True, loo_test=True)**: Perform Random Forest analysis on the feature table.
- **perform_GradientBoost(self, model='XGBoost', group_info=None, annotated_only=True, loo_test=True)**: Perform Gradient Boosting analysis on the feature table.
- **perform_T_Test(self, group_info=None, annotated_only=True)**: Perform T-Test analysis on the feature table.
- **select_biomarker(self, criterion={'PLS_VIP': ['>', 1.5]}, combination='union', annotated_only=True)**: Select biomarkers from the feature table based on given criteria.
- **perform_heatmap(self, biomarker_only=True, group_info=None, hide_xticks=False, hide_ytick=False)**: Perform heatmap visualization of the feature table or biomarker table.
- **perform_molecular_network(self, threshold=0.5, target_compound=None, group_info=None)**: Perform molecular network analysis based on the feature table.
- **perform_enrichment_analysis(self, organism="hsa", pvalue_cutoff=0.05, adj_method="fdr_bh")**: Perform enrichment analysis on the biomarker table.
- **save_project(self, save_path)**: Save the current project to a file.
- **load_project(self, save_path)**: Load a project from a file.

**Attributes:**

- **files**: list, The list of sample names corresponding to the columns in the feature table.
- **feature_table**: DataFrame, Extracted features with all obtained information from the executed methods.
- **feature_table_annotated**: DataFrame, Extracted features which have been annotated.
- **biomarker_table**: DataFrame, Table of selected biomarker.