# Enhancement Analysis

Enrichment analysis is a bioinformatics method used to identify overrepresented biological 
terms or functional categories within a set of genes, proteins, or metabolites of interest. 
It helps researchers gain insights into the biological significance of their experimental 
results by determining whether specific functional annotations or pathways are enriched in 
the dataset compared to what would be expected by chance.

Here's an example code snippet:
    
    # after Feature Detection, Metabolite Annotation and Statistical Analysis
    automs_hpic_feat.select_biomarker(criterion = {'PLS_VIP': ['>', 1.2]})
    automs_hpic_feat.perform_enrichment_analysis(self, organism="hsa", pvalue_cutoff = 0.05, adj_method = "fdr_bh")
    
**Parameters**:

- perform_enrichment_analysis
    - organism (str): The organism for enrichment analysis. Default is "hsa" (human).
    - pvalue_cutoff (float): The p-value cutoff for significant enrichment. Default is 0.05.
    - adj_method (str): The adjustment method for multiple testing correction. Default is "fdr_bh".

