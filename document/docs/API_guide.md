# API docs

This is an overview over all classes available in AutoMS.

## AutoMSData
The AutoMSData class represents an object for handling mass spectrometry data.

**Methods:**

- **__init__(self, ion_mode='positive')**: Initialize the AutoMSData object with an optional ion_mode parameter specifying the ionization mode.
- **load_files(self, data_path)**: Load data files from the specified directory path.
- **find_features(self, min_intensity, mass_inv=1, rt_inv=30, min_snr=3, max_items=50000)**: Find features in the loaded data files using the HPIC algorithm. Parameters include min_intensity for the minimum intensity threshold, mass_inv for the inverse of mass tolerance, rt_inv for the inverse of retention time tolerance, min_snr for the minimum signal-to-noise ratio threshold, and max_items for the maximum number of ion traces to process.


## AutoMSFeature