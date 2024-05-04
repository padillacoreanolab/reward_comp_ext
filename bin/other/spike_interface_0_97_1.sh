conda create -n spike_interface_0_97_1 python=3.10 --yes
conda activate spike_interface_0_97_1
conda install -c conda-forge jupyterlab --yes

# for spike sorting and preprocessing
# https://spikeinterface.readthedocs.io/en/latest/installation.html
pip install spikeinterface[full,widgets]==0.97.1

# For spike sorting
pip install --upgrade mountainsort5