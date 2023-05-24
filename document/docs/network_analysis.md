# Molecular Network Analysis

Molecular networks are powerful tools used in metabolomics to analyze and visualize 
the relationships between metabolites based on their structural similarities or 
chemical associations. These networks provide insights into the connectivity and 
interactions among metabolites, helping to uncover potential pathways, functional 
modules, and biological relationships.

In a molecular network, nodes represent individual metabolites, and edges represent 
connections between metabolites based on their chemical similarity or shared metabolic 
transformations. The construction of molecular networks involves analyzing data such 
as mass spectrometry-based metabolite profiles or chemical databases to identify 
structural similarities or functional relationships between metabolites.

AutoMS provides dynamic graph of global network, including all annotated metabolites,
highlighting the biomarkers if they have been selected by statistical analysis. In this 
plot, each metabolite is a single point.

It also provides sub-network if a specific compound need to be focused on. In this plot, 
metabolites shown as chemical images, accompanied with chemical names and relative concentration 
relationships among groups

Here's an example code snippet:
    
    # after Feature Detection, Metabolite Annotation and Statistical Analysis
    automs_hpic_feat.select_biomarker(criterion = {'PLS_VIP': ['>', 1.2]})
    automs_hpic_feat.perform_molecular_network(threshold = 0.5)
    # or
    automs_hpic_feat.perform_molecular_network(threshold = 0.5, target_compound = 'Stachyose', group_info = group_info)
    
**Parameters**:

- perform_molecular_network
    - threshold (float): The threshold value of similarity for creating the network. Default is 0.5.
    - target_compound (str): The target compound to focus on in the network. Default is None.
    - group_info (dict): The dictionary specifying group information for sample grouping.