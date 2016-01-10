# Installation

As noted from scikit-learn website, it is not recommended to install scypy or numpy using pip. These
packages can be installed using the following command

```shell
sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy 
libatlas-dev libatlas3gf-base
```

Once complete install python packages using
```shell
workon ml-env
pip install -r requirements.txt
```