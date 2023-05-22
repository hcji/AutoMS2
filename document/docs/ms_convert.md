# Converting MS Data Files to mzML

Mass spectrometry (MS) data files acquired from different vendors often come in proprietary formats 
that are not compatible with standard analysis tools. AutoMS support the mzML format only, 
one widely used approach is to convert the vendor-specific data files into the mzML format, 
which is a standardized file format for MS data.
To convert MS data files into mzML using msconvert, follow the steps outlined below:

## Step 1: Install ProteoWizard
Before using msconvert, you need to install the ProteoWizard software suite. Follow these instructions to install it: 
 
- Visit the ProteoWizard website: [ProteoWizard Website](http://proteowizard.sourceforge.net/).  
- Download the appropriate installer for your operating system.  
- Run the installer and follow the on-screen instructions to complete the installation.

## Step 2: Open a Command Prompt or Terminal
Once ProteoWizard is installed, open a command prompt (Windows) or terminal (macOS/Linux) to access the msconvert tool.

## Step 3: Convert MS Data to mzML
To convert the MS data file to mzML, use the following command:

        msconvert <input_file> -o <output_directory> --mzML

Replace *input_file* with the path to the vendor-specific MS data file that you want to convert. Specify the *output_directory* where you want to save the converted mzML file.

For example, to convert a Thermo RAW file named sample.raw to mzML and save it in the current directory, use the command:

        msconvert sample.raw -o . --mzML

## Step 4: Additional Conversion Options
msconvert provides additional options to customize the conversion process. Some common options include:

--filter: Apply data filtering during conversion.  
--mz5: Convert the data to the mz5 format.  
--gzip: Compress the output mzML file using gzip.  
--32-bit: Convert the data to 32-bit floating-point format.  

Refer to the msconvert documentation for a full list of available options and their usage.