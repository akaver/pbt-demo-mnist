# pbt-demo-mnist
Population Based Training demo, MNIST

This is simplified toy demo grown out from my Master thesis work.  
Shows how to use Ray Tune Population Based Training for modifying hyperparameters (my work is/was focused on augmentations scheduling) during training.  

Development work is mostly done using Jetbrains PyCharm Professional and DataSpell.



## Code organization

FashionMNISTDataset.py - FashionMNISTDataset class inherited from torchvision.datasets.FashionMNIST  
overrides __getitem__ to provide nomalization of image.

FashionMNISTLightningDataModule.py - FashionMNISTLightningDataModule inherited from PL LightningDataModule  
Prepares data and provides dataloaders  
Calculates datasets mean and std if not provided in conf.  
Applies augmentations to data in on_before_batch_transfer method. Data is multiplied - original plus every augmented version.  

FashionMNISTLightningModule.py - Pytorch Lightning based neural network. Support logging needed for population based training

FashionMNIST-train.(py|ipynb) - test training FashionMNISTLightningModule without full blown Population Based Training

FashionMNIST-pbt.(py|ipynb) - Population based training (Ray Tune), model is FashionMNISTLightningModule.

view-schedule.(py|ipynb) - view pbt produced hyperparams schedule after pbt training is done (grab the best model id from training)

Actual augmentations are specified in augmentation/augmentation.py

## Useful code blocks for running in Jupyter notebooks
### Install packages
~~~python
import sys
!{sys.executable} -m pip install --upgrade filelock
~~~

### Enable imported source code modules autoreload
https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
~~~~python
%load_ext autoreload
%autoreload 2
~~~~

### Launch tensorboard (only in jupyter notebook - not in jb dataspell)
~~~python
%load_ext tensorboard
%tensorboard --logdir data/lightning_logs/
~~~