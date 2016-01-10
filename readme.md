# Installation

As noted from scikit-learn website, it is not recommended to install scypy or numpy using pip. These
packages can be installed using the following command

```shell
sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy 
liblapack-dev libatlas-dev libatlas3gf-base gfortran
```

[More info on installation](http://www.scipy.org/scipylib/building/linux.html)

According to the official doc, make sure that ATLAS is used to provide the implementation of the 
BLAS and LAPACK linear algebra routines:

```shell
sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
```

Once complete install python packages using
```shell
workon ml-env
pip install -r requirements.txt
```

Update: Conda turns out to be an extremely helpful solution to install all the scipy packages
without much hassle.

## Installation guide for Conda:
 1. Download mini conda from [here](http://conda.pydata.org/miniconda.html)
 2. Install conda: `bash Downloads/Miniconda3-latest-Linux-x86_64.sh`
 3. Create a conda env(much like virtualenv but it's conda!): `conda create -n ml-conda python`
 4. Activate ml-conda env: `source activate ml-conda`
 5. Now install conda packages as 
 ```shell
 conda install pip
 conda install scikit-learn
 conda install matplotlib
 conda install pandas
 ```
