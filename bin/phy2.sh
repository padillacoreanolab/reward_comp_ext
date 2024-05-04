conda create --name phy2 --yes cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python==3.9 qtconsole requests responses scikit-learn scipy traitlets

conda activate phy2
pip install git+https://github.com/cortex-lab/phy.git
