conda create -n spike_interface_0_98_2 python=3.9 --yes
conda activate spike_interface_0_98_2
conda install -c conda-forge jupyterlab --yes

# for spike sorting and preprocessing
# https://spikeinterface.readthedocs.io/en/latest/installation.html
pip install spikeinterface[full,widgets]==0.98.2
# For spike sorting
pip install --upgrade mountainsort5