# Turning on Conda
module load conda

# Creating a new environment
mamba create -y -p ./sleap_1_3_3 -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.3

# Turning on created environment
mamba activate ./sleap_1_3_3

### To use Jupyterlab
mamba install jupyterlab -c conda-forge --yes

### To get the Git repo root directory
mamba install -c conda-forge gitpython --yes

### To read and write Excel files
mamba install -c conda-forge openpyxl --yes

## To analyze h5 files(maybe remove)
mamba install conda-forge::h5py --yes

## To analyze images
mamba install conda-forge::imageio --yes
mamba install conda-forge::pillow --yes

### For better quality plots
mamba install seaborn -c conda-forge --yes

## To run HDBSCAN
mamba install conda-forge::hdbscan --yes

## UMAP projections
mamba install -c conda-forge umap-learn --yes

## Fixing Numba
pip install --upgrade numpy






