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

Dimensionality reduction analysis is a technique used in machine learning and data 
analysis to reduce the dimensionality of a dataset while preserving its important structures 
and patterns. It aims to overcome the curse of dimensionality by transforming high-dimensional 
data into a lower-dimensional space.

**Principal Component Analysis (PCA)**: PCA is a widely used linear dimensionality reduction technique. 
It identifies the directions in the data that capture the maximum variance, known as principal 
components. These components form a new orthogonal coordinate system, where the data can be 
represented with fewer dimensions while retaining most of the original information.

**t-Distributed Stochastic Neighbor Embedding (t-SNE)**: t-SNE is a nonlinear dimensionality 
reduction method that emphasizes the preservation of local structure and relationships 
between data points. It maps high-dimensional data to a lower-dimensional space by 
modeling the similarity between data points in the original space and the lower-dimensional 
space. t-SNE is particularly useful for visualizing clusters and identifying patterns in complex datasets.

**Uniform Manifold Approximation and Projection (UMAP)**: UMAP is a nonlinear dimensionality 
reduction algorithm that preserves both local and global structure. It constructs a 
high-dimensional graph representation of the data and optimizes a low-dimensional graph 
representation that captures the same topological structure. UMAP is known for its 
scalability and ability to preserve complex data relationships.

Here's an example code snippet:

    automs_hpic_feat.perform_dimensional_reduction(group_info = group_info, method = 'PCA', annotated_only = True)
    automs_hpic_feat.perform_dimensional_reduction(group_info = group_info, method = 'tSNE', annotated_only = True)
    automs_hpic_feat.perform_dimensional_reduction(group_info = group_info, method = 'uMAP', annotated_only = True)
    
**Parameters**:

- perform_dimensional_reduction  
    - group_info (dict): The dictionary containing group information for sample grouping.
    - method (str): The dimensional reduction method to use (e.g., 'PCA', 'tSNE', 'uMAP').
    - annotated_only (bool): Whether to use only the annotated feature table for dimensional reduction.
    - args: Additional keyword arguments for the dimensional reduction method.


## T-Test Analysis
T-test analysis is a statistical method used to compare the means of two groups and determine 
if there is a significant difference between them. It is commonly used in hypothesis testing 
to assess whether the difference observed between two sample means is likely to occur due to 
random chance or if it represents a true difference in the population means. 

AutoMS conduct T-Test by *perform_T_Test* function. Multi test correlation is integrated, and 
volcano plot is carry out automatically.

Here's an example code snippet:

    group_info = {'PX_L': ['PX-L-{}'.format(i) for i in range(1,7)],
                  'ZX_L': ['ZX-L-{}'.format(i) for i in range(1,7)]}
    automs_hpic_feat.perform_T_Test(group_info = group_info, , annotated_only = True, alpha=0.05, method='fdr_bh')
    
**Parameters**:

- perform_T_Test  
    - group_info (dict): The dictionary specifying group information for sample grouping.
    - annotated_only (bool): Flag indicating whether to use only the annotated feature table. Default is True.
    - alpha (float): significance level to apply.
    - method (str): multiple testing correction method to apply.
    
    
## Multivariate Statistical Analysis

The multivariate statistical analysis methods, including PLS-DA, RF, and Gradient Boosting, 
provide valuable tools for feature selection, classification, and predictive modeling 
in metabolomics research. They aid in understanding the underlying patterns and discriminating 
features in large and complex metabolomics datasets, facilitating biomarker discovery, 
sample classification, and interpretation of metabolite associations with different conditions or phenotypes.

- *Partial Least Squares Discriminant Analysis (PLS-DA)*: PLS-DA is a supervised multivariate 
technique commonly used in metabolomics to classify samples into predefined groups based 
on their metabolite profiles. It identifies the metabolites that contribute the most to 
the separation between groups and creates a predictive model. PLS-DA combines features of 
both principal component analysis (PCA) and linear regression to maximize the separation 
between groups while considering the relationship between metabolites and sample classes.

- *Random Forest (RF)*: RF is a machine learning algorithm widely applied in metabolomics 
for classification and feature selection. It constructs an ensemble of decision trees 
and combines their predictions to make accurate classifications. RF can handle high-dimensional 
data and identify the most important metabolites for classification, allowing for 
the interpretation of discriminatory features.

- *Gradient Boosting*: Gradient boosting is another machine learning technique commonly 
used in metabolomics, particularly with algorithms like XGBoost or LightGBM. Gradient 
boosting sequentially trains a series of weak models (usually decision trees) by focusing 
on the samples that were not well predicted by previous models. The weak models are 
then combined to create a strong predictive model that can accurately classify samples 
or predict their properties. Gradient boosting can handle complex relationships and 
capture non-linear patterns in metabolomics data.

Here's an example code snippet:

    automs_hpic_feat.perform_PLSDA(group_info = group_info, n_components = 3, n_permutations = 1000, annotated_only = True, loo_test = True, permutation_test = True)
    automs_hpic_feat.perform_GradientBoost(group_info = group_info, annotated_only = True, loo_test = True, model = 'XGBoost', n_estimators = 500)
    automs_hpic_feat.perform_RandomForest(group_info = group_info, annotated_only = True, loo_test = True, n_estimators = 500, max_depth = 10)

**Parameters**:

- perform_PLSDA  
    - group_info (dict): The dictionary specifying group information for sample grouping.
    - n_components (int): The number of components to use. Default is 2.
    - n_permutations (int): The number of permutations to perform for permutation test. Default is 1000.
    - annotated_only (bool): Flag indicating whether to use only the annotated feature table. Default is True.
    - loo_test (bool): Flag indicating whether to perform leave-one-out test. Default is True.
    - permutation_test (bool): Flag indicating whether to perform permutation test. Default is True.
    
- perform_GradientBoost  
    - model (str): The gradient boosting model to use ('XGBoost' or 'LightGBM'). Default is 'XGBoost'.
    - group_info (dict): The dictionary specifying group information for sample grouping.
    - annotated_only (bool): Flag indicating whether to use only the annotated feature table. Default is True.
    - loo_test (bool): Flag indicating whether to perform leave-one-out test. Default is True.
    - args: Additional arguments to be passed to the Gradient Boosting analysis.

- perform_RandomForest
    - group_info (dict): The dictionary specifying group information for sample grouping.
    - annotated_only (bool): Flag indicating whether to use only the annotated feature table. Default is True.
    - loo_test (bool): Flag indicating whether to perform leave-one-out test. Default is True.
    - args: Additional arguments to be passed to the Random Forest analysis.


## HeatMap of Biomarkers

A heatmap of metabolite biomarkers is a graphical representation that visualizes the relative 
abundance or expression levels of metabolites across different samples or conditions. 
It provides a comprehensive overview of the patterns and trends in metabolite profiles, 
highlighting the metabolites that show significant differences or associations with specific groups or conditions.

In a heatmap, each row represents a metabolite, and each column represents a sample or condition. 
The intensity of color or shading in each cell of the heatmap reflects the abundance or expression 
level of the corresponding metabolite in the respective sample or condition. Typically, a 
color scale is used, where higher values are represented by darker or brighter colors, and 
lower values are represented by lighter or muted colors.

Here's an example code snippet:

    # This step should after perform T-Test analysis or/and multivariate statistical analysis
    automs_hpic_feat.select_biomarker(criterion = {'PLS_VIP': ['>', 1.5], 'RF_VIP': ['>', 0.2]}, combination = 'union')
    automs_hpic_feat.perform_heatmap(group_info = group_info, hide_xticks = False, hide_ytick = False)

**Parameters**:

- select_biomarker
    - criterion (dict): The dictionary specifying the criteria for biomarker selection.
    - combination (str): The combination method for multiple criteria ('union' or 'intersection'). Default is 'union'.
    - annotated_only (bool): Flag indicating whether to use only the annotated feature table. Default is True.

- perform_heatmap
    - biomarker_only (bool): Flag indicating whether to use only the biomarker table. Default is True.
    - group_info (dict): The dictionary specifying group information for sample grouping.
    - hide_xticks (bool): Flag indicating whether to hide the x-axis tick labels. Default is False.
    - hide_ytick (bool): Flag indicating whether to hide the y-axis tick labels. Default is False.
