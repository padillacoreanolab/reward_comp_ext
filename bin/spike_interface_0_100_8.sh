# Turning on Conda
module load conda

# Creating a new environment
mamba create -p ./spike_interface_0_100_8 python=3.10 --yes
# Turning on created environment
mamba activate ./spike_interface_0_100_8

### To use GPU
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

### To do GPU calculations with Numpy
mamba install -c conda-forge cupy --yes

### To use Jupyterlab
mamba install jupyterlab -c conda-forge --yes

### To get the Git repo root directory
mamba install -c conda-forge gitpython --yes

### To read and write Excel files
mamba install -c conda-forge openpyxl --yes

### To use spikeinterface
pip install spikeinterface[full,widgets]==0.100.8
pip install --upgrade mountainsort5==0.5.6

### To calculate spectral metrics
mamba install -c edeno spectral_connectivity --yes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# Below causes issues with cannot import name 'ImplementsArrayReduce' from partially initialized module 'xarray.core.common' (most likely due to a circular import)

### To make statistical models
mamba install -c conda-forge statsmodels --yes

### To calculate Spike-LFP Coupling
mamba install -c conda-forge astropy --yes

### For better quality plots
mamba install seaborn -c conda-forge --yes

## UMAP projections
mamba install -c conda-forge umap-learn --yes

## To analyze h5 files(maybe remove)
mamba install conda-forge::h5py --yes

## To analyze images
mamba install conda-forge::imageio --yes
# (maybe remove)
mamba install conda-forge::pillow --yes

## To run HDBSCAN
mamba install conda-forge::hdbscan --yes

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~










# Installing the necessary packages

### To use GPU
mamba install cudatoolkit=12.5 pytorch=1.12.1=gpu_cuda* -c pytorch --yes

### To use Jupyterlab
mamba install jupyterlab -c conda-forge --yes

### To use spikeinterface
pip install spikeinterface[full,widgets]==0.100.5
pip install --upgrade mountainsort5==0.5.6

### To calculate spectral metrics
mamba install -c edeno spectral_connectivity --yes

### To calculate power with CWT
pip install fCWT==0.1.18

### To make statistical models
mamba install -c conda-forge statsmodels --yes

### To calculate Spike-LFP Coupling
mamba install -c conda-forge astropy --yes

### To do GPU calculations with Numpy
mamba install -c conda-forge cupy --yes

### For better quality plots
mamba install seaborn -c conda-forge --yes

### To get the Git repo root directory
mamba install -c conda-forge gitpython --yes

### To read and write Excel files
mamba install -c conda-forge openpyxl --yes

### To use fft on fcwt
mamba install -c conda-forge fftw --yes