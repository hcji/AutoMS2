# API Guide

This is an overview over all classes available in AutoMS.

## AutoMSData
The AutoMSData class represents an object for handling mass spectrometry data.

**Methods:**

- **__init__(self, ion_mode='positive')**: Initialize the AutoMSData object with an optional ion_mode parameter specifying the ionization mode.
- **load_files(self, data_path)**: Load data files from the specified directory path.
- **find_features(self, min_intensity, mass_inv=1, rt_inv=30, min_snr=3, max_items=50000)**: Find features in the loaded data files using the HPIC algorithm. 

**Attributes:**

- **data_path**: 
- **ion_mode**: 
- **peaks**: 
- **feature_table**: 


## AutoMSFeature