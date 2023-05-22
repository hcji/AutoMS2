# Getting Started with the AutoMS

following example code demonstrates the usage of the AutoMS software for feature extraction, 
feature matching, library searching, and data analysis in mass spectrometry-based metabolomics. 
The code showcases various steps involved in the processing and analysis of metabolomics 
data using the AutoMS package.
 
    from AutoMS import automs

    # Feature extraction for positive ion mode
    automs_hpic_pos = automs.AutoMSData(ion_mode='positive')
    automs_hpic_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
    automs_hpic_pos.find_features(min_intensity=20000, max_items=100000)

    # Feature matching for positive ion mode
    automs_hpic_pos.match_features()
    automs_hpic_pos.match_features_with_ms2()

    # Library searching for positive ion mode
    automs_hpic_pos.search_library("Library/references_spectrums_positive.pickle")

    # Save project for positive ion mode
    automs_hpic_pos.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")

    # Feature extraction for negative ion mode
    automs_hpic_neg = automs.AutoMSData(ion_mode='negative')
    automs_hpic_neg.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/NEG")
    automs_hpic_neg.find_features(min_intensity=20000, max_items=100000)

    # Feature matching for negative ion mode
    automs_hpic_neg.match_features()
    automs_hpic_neg.match_features_with_ms2()
    
    # Library searching for negative ion mode
    automs_hpic_neg.search_library("Library/references_spectrums_negative.pickle")
    
    # Save project for negative ion mode
    automs_hpic_neg.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")
    
    # Group information
    qc_samples = ['QC-{}'.format(i) for i in range(1,6)]
    group_info = {
        'QC': ['QC-{}'.format(i) for i in range(1,6)],
        'PX_L': ['PX-L-{}'.format(i) for i in range(1,7)],
        'PX_S': ['PX-S-{}'.format(i) for i in range(1,7)],
        'ZX_L': ['ZX-L-{}'.format(i) for i in range(1,7)],
        'ZX_S': ['ZX-S-{}'.format(i) for i in range(1,7)],
        'NX_L': ['NX-L-{}'.format(i) for i in range(1,7)],
        'NX_S': ['NX-S-{}'.format(i) for i in range(1,7)]
    }

    # Load and export features for positive ion mode
    automs_hpic_pos = automs.AutoMSData(ion_mode='positive')
    automs_hpic_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")
    automs_hpic_pos = automs_hpic_pos.export_features()
    
    # Load and export features for negative ion mode
    automs_hpic_neg = automs.AutoMSData(ion_mode='negative')
    automs_hpic_neg.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")
    automs_hpic_neg = automs_hpic_neg.export_features()
    
    # Merge positive and negative features
    automs_hpic_feat = automs.AutoMSFeature()
    automs_hpic_feat.append_feature_table(automs_hpic_pos)
    automs_hpic_feat.append_feature_table(automs_hpic_neg)
    
    # Preprocessing and refining annotated table
    automs_hpic_feat.preprocessing(
        impute_method='KNN',
        outlier_threshold=3,
        rsd_threshold=0.3,
        min_frac=0.5,
        qc_samples=qc_samples,
        group_info=group_info
    )
    automs_hpic_feat.refine_annotated_table()
    
    # Perform dimensional reduction and analysis
    automs_hpic_feat.perform_dimensional_reduction(
        group_info=group_info,
        method='PCA',
        annotated_only=False
    )
    automs_hpic_feat.perform_PLSDA(group_info=group_info, n_components=3)
    automs_hpic_feat.perform_RandomForest(group_info=group_info)
    
    # Select biomarkers and generate heatmap
    automs_hpic_feat.select_biomarker(
        criterion={'PLS_VIP': ['>', 1.2], 'RF_VIP': ['>', 0.15]},
        combination='intersection'
    )
    automs_hpic_feat.perform_heatmap(
        group_info=group_info,
        hide_xticks=False,
        hide_ytick=False
    )


The provided code demonstrates the usage of AutoMS for feature extraction, 
matching, library searching, and subsequent data analysis steps. It covers both 
positive and negative ion modes, showcases project loading and saving, and 
includes preprocessing, dimensional reduction, and biomarker selection. 
These steps enable the processing and analysis of mass spectrometry-based 
metabolomics data using the AutoMS software.