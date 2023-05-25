<style>
pre {
  overflow-y: auto;
  max-height: 300px;
}
</style>


# Annotation of Extracted Features

Feature annotation is the process of assigning putative identities or annotations 
to detected features in metabolomics experiments. It involves associating the detected 
features with known or predicted metabolites present in databases or reference libraries. 

## Annotation with Library Search

AutoMS provides library search algorithm based on [SpectralEntropy](https://github.com/YuanyueLi/SpectralEntropy) 
package. 43 different spectral similarity algorithms can be used for MS/MS spectral comparison.

Here's an example code snippet:
    
    """
    # Feature Extraction and Matching
    automs_hpic_pos = automs.AutoMSData(ion_mode='positive')
    automs_hpic_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
    automs_hpic_pos.find_features(min_intensity=20000, max_items=100000)
    automs_hpic_pos.match_features()
    """
    
    # Match features with corresponding MS/MS spectra
    automs_hpic_pos.match_features_with_ms2()
    
    # Search a library for metabolite annotation based on the feature table
    automs_hpic_pos.search_library("Library/references_spectrums_positive.pickle")

**Parameters**:

- match_features_with_ms2    
    - mz_tol (float): The m/z tolerance for matching features with spectra. Default is 0.01.
    - rt_tol (float): The retention time tolerance for matching features with spectra. Default is 15.
    
- search_library  
    - lib_path (str): The path to the library file.
    - method (str): The method for library search. Default is 'entropy'.
    - ms1_da (float): The m/z tolerance for matching MS1 masses. Default is 0.01.
    - ms2_da (float): The m/z tolerance for matching MS2 masses. Default is 0.05.
    - threshold (float): The annotation confidence threshold. Default is 0.5.
    
In this example, the code creates an AutoMSData object with the ion mode set to 'positive'. 
The data files are then loaded from the specified directory. Features are found in the loaded 
data using the specified parameters. Feature matching is performed, followed by feature 
matching with MS2 spectra. Finally, the feature annotation step is carried out using a library 
stored in the specified pickle file ("Library/references_spectrums_positive.pickle").  
  
Processed public libraries ready for AutoMS can be accessed via [MS/MS Library](https://hcji.github.io/AutoMS2/library/) page. 


**Supported similarity method**:

    "entropy": Entropy distance
    "unweighted_entropy": Unweighted entropy distance
    "euclidean": Euclidean distance
    "manhattan": Manhattan distance
    "chebyshev": Chebyshev distance
    "squared_euclidean": Squared Euclidean distance
    "fidelity": Fidelity distance
    "matusita": Matusita distance
    "squared_chord": Squared-chord distance
    "bhattacharya_1": Bhattacharya 1 distance
    "bhattacharya_2": Bhattacharya 2 distance
    "harmonic_mean": Harmonic mean distance
    "probabilistic_symmetric_chi_squared": Probabilistic symmetric χ2 distance
    "ruzicka": Ruzicka distance
    "roberts": Roberts distance
    "intersection": Intersection distance
    "motyka": Motyka distance
    "canberra": Canberra distance
    "baroni_urbani_buser": Baroni-Urbani-Buser distance
    "penrose_size": Penrose size distance
    "mean_character": Mean character distance
    "lorentzian": Lorentzian distance
    "penrose_shape": Penrose shape distance
    "clark": Clark distance
    "hellinger": Hellinger distance
    "whittaker_index_of_association": Whittaker index of association distance
    "symmetric_chi_squared": Symmetric χ2 distance
    "pearson_correlation": Pearson/Spearman Correlation Coefficient
    "improved_similarity": Improved Similarity
    "absolute_value": Absolute Value Distance
    "dot_product": Dot-Product (cosine)
    "dot_product_reverse": Reverse dot-Product (cosine)
    "spectral_contrast_angle": Spectral Contrast Angle
    "wave_hedges": Wave Hedges distance
    "cosine": Cosine distance
    "jaccard": Jaccard distance
    "dice": Dice distance
    "inner_product": Inner Product distance
    "divergence": Divergence distance
    "avg_l": Avg (L1, L∞) distance
    "vicis_symmetric_chi_squared_3": Vicis-Symmetric χ2 3 distance
    "ms_for_id_v1": MSforID distance version 1
    "ms_for_id": MSforID distance
    "weighted_dot_product": Weighted dot product distance"
    

## How to Build a Library

library should be stored in the specified pickle file, which is a list of matchms::Spectrum object.

Here's an example code snippet of how to build library with GNPS data. GNPS data can be downloand at [url](https://external.gnps2.org/gnpslibrary)

    import os 
    import pickle 
    import numpy as np 
    from tqdm import tqdm 
    from matchms.importing import load_from_mgf

    # Setting the path to the data directory
    path_data = os.path.join('D:/All_MSDatabase/GNPS_all')

    # Creating the full path to the MGF file
    filename = os.path.join(path_data, 'ALL_GNPS.mgf')
    
    # Loading spectrums from the MGF file using the 'load_from_mgf' function and creating a list of spectrums
    spectrums = [s for s in tqdm(load_from_mgf(filename))]
    
    from matchms.filtering import default_filters  # Importing a function from the 'matchms.filtering' module
    from matchms.filtering import add_parent_mass, derive_adduct_from_name  # Importing functions from the 'matchms.filtering' module
    
    # Defining a function to apply filters to a spectrum
    def apply_filters(s):
        s = default_filters(s)  # Applying default filters to the spectrum
        s = derive_adduct_from_name(s)  # Deriving the adduct from the spectrum's name
        s = add_parent_mass(s, estimate_from_adduct=True)  # Adding the parent mass to the spectrum
        return s
    
    # Applying filters to the spectrums and creating a new list of filtered spectrums
    spectrums = [apply_filters(s) for s in tqdm(spectrums) if s is not None]
    
    # Saving the filtered spectrums as a NumPy array
    np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)
    
    
    from matchms.filtering import harmonize_undefined_inchikey, harmonize_undefined_inchi, harmonize_undefined_smiles  # Importing functions from the 'matchms.filtering' module
    from matchms.filtering import repair_inchi_inchikey_smiles  # Importing a function from the 'matchms.filtering' module
    
    # Defining a function to clean the metadata of a spectrum
    def clean_metadata(s):
        s = harmonize_undefined_inchikey(s)  # Harmonizing undefined InChI keys in the spectrum
        s = harmonize_undefined_inchi(s)  # Harmonizing undefined InChI in the spectrum
        s = harmonize_undefined_smiles(s)  # Harmonizing undefined SMILES in the spectrum
        s = repair_inchi_inchikey_smiles(s)  # Repairing InChI, InChI key, and SMILES in the spectrum
        return s
    
    # Cleaning the metadata of the spectrums and creating a new list of spectrums with cleaned metadata
    spectrums = [clean_metadata(s) for s in tqdm(spectrums) if s is not None]
    
    # Saving the spectrums with cleaned metadata as a NumPy array
    np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)
    
    
    from matchms.filtering import derive_inchi_from_smiles, derive_smiles_from_inchi  # Importing functions from the 'matchms.filtering' module
    from matchms.filtering import derive_inchikey_from_inchi  # Importing a function from the 'matchms.filtering' module
    
    # Defining a function to further clean the metadata of a spectrum
    def clean_metadata2(s):
        s = derive_inchi_from_smiles(s)  # Deriving InChI from SMILES in the spectrum
        s = derive_smiles_from_inchi(s)  # Deriving SMILES from InChI in the spectrum
        s = derive_inchikey_from_inchi(s)  # Deriving InChI key from InChI in the spectrum
        return s
    
    # Further cleaning the metadata of the spectrums and creating a new list of spectrums with further cleaned metadata
    spectrums = [clean_metadata2(s) for s in tqdm(spectrums) if s is not None]
    
    # Saving the spectrums with further cleaned metadata as a NumPy array
    np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)
    
    
    # Looping over each spectrum and modifying the compound name
    for spectrum in tqdm(spectrums):
        name_original = spectrum.get("compound_name")  # Getting the original compound name from the spectrum
        name = name_original.replace("F dial M", "")  # Removing "F dial M" from the compound name
    
        # Remove last word if likely not correct
        if name.split(" ")[-1] in ["M", "M?", "?", "M+2H/2", "MS34+Na", "M]", "Cat+M]", "Unk", "--"]:
            name = " ".join(name.split(" ")[:-1]).strip()  # Removing the last word from the compound name
    
        if name != name_original:
            print(f"Changed compound name from {name_original} to {name}.")  # Printing the changed compound name
            spectrum.set("compound_name", name)  # Setting the modified compound name in the spectrum
    
    
    # Looping over each spectrum and modifying the ion mode
    for spec in spectrums:
        if spec.get("adduct") in ['[M+CH3COO]-/[M-CH3]-', '[M-H]-/[M-Ser]-', '[M-CH3]-']:
            if spec.get("ionmode") != "negative":
                spec.set("ionmode", "negative")  # Setting the ion mode to "negative" if specific adducts are present
    
    
    from matchms.filtering import normalize_intensities  # Importing a function from the 'matchms.filtering' module
    from matchms.filtering import require_minimum_number_of_peaks  # Importing a function from the 'matchms.filtering' module
    from matchms.filtering import select_by_mz  # Importing a function from the 'matchms.filtering' module
    
    # Defining a function to post-process a spectrum
    def post_process(s):
        s = normalize_intensities(s)  # Normalizing the intensities of the spectrum
        s = select_by_mz(s, mz_from=10.0, mz_to=1000)  # Selecting peaks within a specific m/z range
        s = require_minimum_number_of_peaks(s, n_required=5)  # Requiring a minimum number of peaks in the spectrum
        return s
    
    # Post-processing the spectrums and creating a new list of post-processed spectrums
    spectrums = [post_process(s) for s in tqdm(spectrums)]
    
    # Saving the post-processed spectrums as a NumPy array
    np.save(os.path.join(path_data, 'preprocessed_spectrums.npy'), spectrums)
    
    spectrums = [s for s in spectrums if s is not None]  # Filtering out any None values from the spectrums
    
    spectrums_positive = []  # Creating an empty list for positive ion mode spectrums
    spectrums_negative = []  # Creating an empty list for negative ion mode spectrums
    
    # Looping over each spectrum and categorizing them based on ion mode
    for i, spec in enumerate(spectrums):
        if spec.get("ionmode") == "positive":
            spectrums_positive.append(spec)  # Adding the spectrum to the positive ion mode list
        elif spec.get("ionmode") == "negative":
            spectrums_negative.append(spec)  # Adding the spectrum to the negative ion mode list
        else:
            print(f"No ionmode found for spectrum {i} ({spec.get('ionmode')})")  # Printing a message if no ion mode is found
    
    # Pickling the negative ion mode spectrums and saving them as a pickle file
    pickle.dump(spectrums_negative, open(os.path.join(path_data, 'ALL_GNPS_220601_negative_cleaned.pickle'), "wb"))
    
    # Pickling the positive ion mode spectrums and saving them as a pickle file
    pickle.dump(spectrums_positive, open(os.path.join(path_data, 'ALL_GNPS_220601_positive_cleaned.pickle'), "wb"))
    

## Annotation with DeepMASS
    
**comming soon**


