conda create -n spike_interface_0_100_6 python=3.10 --yes
conda activate spike_interface_0_100_6
conda install -c conda-forge jupyterlab --yes

# for spike sorting and preprocessing
# https://spikeinterface.readthedocs.io/en/latest/installation.html
pip install spikeinterface[full,widgets]==0.100.6
# For power calculations
conda install -c edeno spectral_connectivity --yes
# For spike sorting
pip install --upgrade mountainsort5