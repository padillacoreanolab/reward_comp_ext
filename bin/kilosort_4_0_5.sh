conda create --name kilosort_4_0_5 python=3.9 --yes
conda activate kilosort_4_0_5
conda install -c conda-forge jupyterlab --yes

# for spike sorting and preprocessing
# https://spikeinterface.readthedocs.io/en/latest/installation.html
pip install spikeinterface[full,widgets]==0.100.6

# For spike sorting
pip install kilosort==4.0.5
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia --yes

# Follow the suggestion here to make sure that all channels show up
# https://github.com/SpikeInterface/spikeinterface/issues/2237#issuecomment-1822757172