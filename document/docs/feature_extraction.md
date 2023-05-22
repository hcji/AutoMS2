# Extraction of Features from Raw Data

Metabolomics is a rapidly growing field that aims to comprehensively study and analyze 
the small molecule metabolites present in biological samples. Mass spectrometry (MS) is 
a widely used technique in metabolomics due to its high sensitivity, resolution, and ability 
to detect a wide range of metabolites.  

## Feature Extraction with AutoMS

AutoMS provides HPIC as the default algorithm of feature extraction. Hierarchical Density-Based 
Spatial Clustering of Applications with Noise (HDBSCAN) was utilized to extract Potential Ion 
Channels (PICs) from LC to MS data sets. Metabolites typically produce densely populated and 
continuous ions in both the m/z (mass-to-charge ratio) and elution time dimensions. 
By employing HDBSCAN, ions belonging to the same metabolite can be grouped together, 
eliminating the need for defining a specific m/z tolerance.

Here's an example code snippet:

    from AutoMS import automs

    # Instantiate AutoMS for positive ion mode
    automs_hpic_pos = automs.AutoMSData(ion_mode='positive')

    # Load data files
    automs_hpic_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")

    # Perform feature extraction
    automs_hpic_pos.find_features(min_intensity=20000, max_items=100000)


**Parameters**:

- AutoMSData    
    - ion_mode (str): The ionization mode. Default is 'positive'.  
    
- load_files  
    - data_path (str): The path to the directory containing the data files.  
    
- find_features  
    - min_intensity (int): The minimum intensity threshold for peak detection. Suggest: Q-TOF 1000-3000; Orbitrap 10000-30000.
    - mass_inv (int): The inverse of mass tolerance for clustering ions of the same metabolite. Default is 1.
    - rt_inv (int): The inverse of retention time tolerance for clustering ions of the same metabolite. Default is 30.
    - min_snr (int): The minimum signal-to-noise ratio threshold for peak detection. Default is 3.
    - max_items (int): The maximum number of ion traces to process. Default is 50000.


## Evaluating Peak Quality of Features
AutoMS employs a deep learning-based denoising autoencoder to grasp the common characteristics 
of chromatographic peaks, and predict noisededucted peaks from the original peak profiles. 
By comparing the difference before and after processed, it scores the peak quality continuously and precisely.

Here's an example code snippet:

    automs_hpic_pos.evaluate_features()


## Matching Features Across Samples
Feature matching is a crucial step in the analysis of mass spectrometry-based metabolomics data. 
It involves comparing and aligning features detected across multiple samples or datasets to identify 
matching features and establish their correspondence.

In metabolomics, features typically represent specific molecular entities such as metabolites.
 These features are characterized by their mass-to-charge ratio (m/z) and retention time (RT), 
 which are important parameters for their identification and quantification.

Feature matching aims to find corresponding features across different samples or datasets by 
comparing their m/z and RT values. This process allows for the identification of consistent features 
that represent the same metabolite across various experimental conditions or biological samples.

Here's an example code snippet:

    automs_hpic_pos.match_features(method = 'simple', mz_tol = 0.01, rt_tol = 20, min_frac = 0.5)
    
**Parameters**:

- match_features
    - method (str): The feature matching method to use. Default is 'simple' (only support at present).
    - mz_tol (float): The mass-to-charge ratio tolerance for feature matching. Default is 0.01.
    - rt_tol (float): The retention time tolerance for feature matching. Default is 20 (seconds).
    - min_frac (float): The minimum fraction of samples that should have a feature for it to be considered. Default is 0.5.


## Load Feature Extraction Results of MS-DIAL

MS-DIAL is a powerful software tool designed for mass spectrometry-based metabolomics data analysis. 
It offers a comprehensive set of functionalities, including feature extraction, alignment, 
and quantification of metabolites. AutoMS can load the feature extraction results of MS-DIAL directly.
By loading the MS-DIAL feature extraction results into the AutoMS software, you can access and further 
analyze the extracted features and their associated information for downstream processing and interpretation.

Here's an example code snippet:

    from AutoMS import automs
    
    # Instantiate AutoMS for positive ion mode
    automs_msdial_pos = automs.AutoMS(ion_mode='positive')
    
    # Load data files
    automs_msdial_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
    
    # Load MS-DIAL feature extraction results
    automs_msdial_pos.load_msdial("E:/Data/Guanghuoxiang/MSDIAL_processing/Positive_Height_0_2023423819.txt")


**Parameters**:

- load_msdial:
    - msdial_path (str): The path to the MS-DIAL feature extraction result file.