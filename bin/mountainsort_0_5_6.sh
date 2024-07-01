conda create -n mountainsort_0_5_6  python=3.9 --yes
conda activate mountainsort_0_5_6
conda install -c conda-forge jupyterlab --yes

# for spike sorting and preprocessing
# https://spikeinterface.readthedocs.io/en/latest/installation.html
pip install spikeinterface[full,widgets]==0.98.2
# For spike sorting
pip install --upgrade mountainsort5==0.5.6

conda install conda-forge::gitpython
pip install openpyxl==3.1.2

pip install umap-learn
conda install conda-forge::opencv
conda install anaconda::numpy