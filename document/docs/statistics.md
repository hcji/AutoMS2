# Statistical Analysis

In metabolomics, the statistical analysis of features plays a crucial role in extracting meaningful 
insights from complex datasets. It aims to understand the metabolic changes and patterns associated 
with various physiological conditions, diseases, and environmental factors. AutoMS provides the key 
aspects of statistical analysis in metabolomics. 

Statistical analysis in AutoMS is performed on *AutoMSFeature* class, which is obtained from *export_features* 
of *AutoMSData* class.

Here's an example code snippet:

    automs_hpic_pos = automs.AutoMSData(ion_mode='positive')
    automs_hpic_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")
    automs_hpic_pos = automs_hpic_pos.export_features()

Here, *automs_hpic_pos* is obtained and saved followed the *feature_extraction* steps.


## Data Combination

Data combination in metabolomics refers to the integration of metabolomics datasets obtained from 
different ionization modes, such as positive and negative modes. Combining data from multiple 
ionization modes can provide a more comprehensive view of the metabolome and improve the coverage 
and reliability of metabolite identification.

AutoMS achieves this by *append_feature_table* function, Here's an example code snippet:

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


## Preprocessing

Preprocessing involves various steps to transform raw metabolomics data into a suitable format for analysis. 
It aims to reduce noise, correct systematic biases, and normalize the data. The preprocessing steps include:

- Data cleaning: Removing missing values and outliers.
- Imputation: Estimating missing values based on statistical methods.
- Data normalization: Scaling the data to account for differences in sample concentration or instrument response.
- Correlation Evaluation: Calculate correlations between QC samples, evaluating the experimental consistency.

Here's an example code snippet:
    
    # Information of samples
    qc_samples = ['QC-{}'.format(i) for i in range(1,6)]
    group_info = {
        'QC': ['QC-{}'.format(i) for i in range(1,6)],       # QC group
        'PX_L': ['PX-L-{}'.format(i) for i in range(1,7)],   # Leaf of PX
        'PX_S': ['PX-S-{}'.format(i) for i in range(1,7)],   # Stem of PX
        'ZX_L': ['ZX-L-{}'.format(i) for i in range(1,7)],   # Leaf of ZX
        'ZX_S': ['ZX-S-{}'.format(i) for i in range(1,7)],   # Stem of ZX
        'NX_L': ['NX-L-{}'.format(i) for i in range(1,7)],   # Leaf of NX
        'NX_S': ['NX-S-{}'.format(i) for i in range(1,7)]    # Stem of NX
    }
    
    # Preprocessing
    automs_hpic_feat.preprocessing(
        impute_method='KNN',
        outlier_threshold=3,
        rsd_threshold=0.3,
        min_frac=0.5,
        qc_samples=qc_samples,
        group_info=group_info
    )
    
    # Refine annotated table and remove repetitive annotations
    automs_hpic_feat.refine_annotated_table()


**Parameters**:

- preprocessing  
    - impute_method (str): The imputation method to use.
    - outlier_threshold (float): The threshold for outlier removal.
    - rsd_threshold (float): The threshold for relative standard deviation (RSD) filtering.
    - min_frac (float): The minimum fraction of non-missing values required for a feature to be retained.
    - qc_samples (list): The list of QC sample names.
    - group_info (dict): The dictionary containing group information for sample grouping.
    - args: Additional keyword arguments for the preprocessing method.
    
**Supported similarity method**:

    'Low value': The missing values are imputed with a low value.
    'Mean': The mean imputation method replaces missing values with the mean value of the corresponding feature.
    'Median': Similar to mean imputation, the median imputation method replaces missing values with the median value of the corresponding feature.
    'KNN': K-nearest neighbors (KNN) imputation imputes missing values by considering the values of the k-nearest neighbors in the feature space.
    'Iterative RF': Iterative Random Forest (RF) imputation is an iterative imputation method. It builds a random forest model to predict the missing values based on other features.
    'Iterative BR': Iterative Bayesian Ridge (BR) imputation is another iterative imputation method. It uses a Bayesian Ridge regression model to estimate the missing values based on other features.
    'Iterative SVR': Iterative Support Vector Regression (SVR) imputation is an iterative method that uses support vector regression to predict missing values. It builds a regression model using non-missing values as the training data and predicts the missing values.


## Dimensional Reduction Analysis




