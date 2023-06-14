# Welcome to AutoMS

## Summary

AutoMS is an open-source Python library for mass spectrometry, specifically for 
the analysis of metabolomics data in Python. AutoMS is available for Windows, Linux and macOS.

AutoMS is a comprehensive software tool designed to facilitate metabolomics data analysis. 
It offers a range of functionalities that aid in the processing and interpretation of metabolomics 
data. Here is a detailed summary of each module within AutoMS:

[![1.png](https://i.postimg.cc/Jzty8SL1/1.png)](https://postimg.cc/4Yk4bLyM)

- Feature Detection:
This module focuses on identifying and extracting relevant features from raw metabolomics data. 
It employs advanced algorithms to detect peaks or signals representing metabolites within mass 
spectrometry or chromatography data.

- Metabolite Annotation:
The metabolite annotation module plays a crucial role in identifying the detected features. 
AutoMS compares the experimental data against reference spectral libraries, such as the MassBank or GNPS libraries. 
By performing spectral matching, it enables the identification and confirmation of metabolites based 
on their characteristic mass spectra.

- Statistical Analysis:
The statistical analysis module within AutoMS offers a suite of powerful statistical methods 
tailored for metabolomics data. It allows researchers to perform various statistical tests, 
including t-tests, multivariate analysis, and machine learning analysis. This facilitates the 
identification of significant differences between groups or conditions, enabling valuable 
insights into metabolic changes.

- Molecular Network:
AutoMS incorporates a molecular network analysis module, which utilizes network-based algorithms 
to explore the relationships between metabolites based on their structural similarities and/or 
shared biosynthetic pathways. This module aids in the elucidation of complex metabolic networks 
and the discovery of potential biomarkers or key metabolites.

- Enhancement Analysis:
The enhancement analysis module focuses on identifying factors or conditions that lead to the 
alteration of specific metabolic pathways or biological functions. It assesses the enrichment 
of metabolites associated with particular pathways or gene sets, providing insights into the 
underlying mechanisms driving metabolic changes.


## Documentation

- [Feature Extraction](https://hcji.github.io/AutoMS2/feature_extraction/)
- [Feature Annotation](https://hcji.github.io/AutoMS2/feature_annotation/)
- [Statistical Analysis](https://hcji.github.io/AutoMS2/statistics/)
- [Network Analysis](https://hcji.github.io/AutoMS2/network_analysis/)
- [Enhancement Analysis](https://hcji.github.io/AutoMS2/enhancement_analysis/)


## Sources on GitHub
[AutoMS2 GitHub](https://github.com/hcji/AutoMS2)

## Changelog

- verision 1.0: *first release* at 20YY.MM.DD

## Citations
**Main citations** (papers of this software)

- Hongchao Ji, Jing Tian. Deep denoising autoencoder-assisted continuous scoring of peak quality in high-resolution LC−MS data. Chemometrics and Intelligent Laboratory Systems 2022, 104694.

**Additional citations** (algorithms integrated in this software)

- Hongchao Ji, Fanjuan Zeng, Yamei Xu, et al. KPIC2: An Effective Framework for Mass Spectrometry-Based Metabolomics Using Pure Ion Chromatograms. Analytical Chemistry 2017, 89 (14), 7631–7640. 
- Hongchao Ji, Yamei Xu, Hongmei Lu, Zhimin Zhang. Deep MS/MS-Aided Structural-Similarity Scoring for Unknown Metabolite Identification. Analytical Chemistry 2019, 91 (9), 5629–5637.
- Huimin Zhu, Yi Chen, Cha Liu, et al. Feature Extraction for LC–MS via Hierarchical Density Clustering. Chromatographia 2019, 10(82): 1449-1457.
- Zhimin Zhang, Xia Tong, Ying Peng, et al. Multiscale peak detection in wavelet space. The Analyst 2015, 23(140): 7955-7964.
- Yuanyue Li, Tobias Kind, Jacob Folz, et al. Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification. Nature Methods 12(8): 1524-1531.
- Florian Huber, Stefan Verhoeven, Christiaan Meijer, et al. matchms - processing and similarity evaluation of mass spectrometry data. Journal of Open Source Software 52(5):2411.
