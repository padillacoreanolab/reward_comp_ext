conda create -n dcsfa_nmf_env  python=3.9 --yes
conda activate dcsfa_nmf_env

conda install -c conda-forge jupyterlab --yes
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes

conda install conda-forge::scikit-learn --yes
conda install conda-forge::matplotlib --yes
conda install conda-forge::numpy --yes
pip install pickle-mixin 
pip install beta-divergence-metrics
conda install conda-forge::tqdm --yes
conda install conda-forge::pandas --yes