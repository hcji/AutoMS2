# Installation Guide
Below the workflow of installation is presented. Install the dependencies first, 
Then, install AutoMS.

## Install dependencies

Please follow the steps:

### Step 1: Install Anaconda
Visit the [Anaconda](https://www.anaconda.com/products/individual) website, 
Download the appropriate installer for your operating system, and run the installer 
and follow the on-screen instructions to complete the installation.

### Step 2: Create a new environment
Open the command prompt, and run the following command to create a new 
environment with the specific Python version:

    conda create --name automs python=3.8.16

Activate the newly created environment by running the following command:

    conda activate automs
    
### Step 3: Install dependencies

Use conda install to install the following packages:

    conda install -c conda-forge hdbscan==0.8.29


## Install AutoMS

You can either install from GitHub or PyPI:

### Install from GitHub:

Running the following command:

    pip install git+https://github.com/hcji/AutoMS2.git
    
### Install from PyPI

Comming soon...