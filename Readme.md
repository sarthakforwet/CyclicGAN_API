# Introduction
## CyclicGAN

This API is an implementation of the `CyclicGAN` been presented in the [original paper](https://arxiv.org/abs/1703.10593).

CylicGAN is a special type of GAN network which works with Unpaired Data to extract mapping between the
two domains (X and Y). We generally have two Generator networks and two Discriminator networks to criticize
the work of the Generators. Apart from the `Adversarial loss`, we also have `cyclic consistency loss` which
makes sure that any mapping of the image from the domain `X` to domain `Y` has the same reverse mapping.

This API aims to makes it easy for developers to train and infer from a CyclicGAN model.

# Dependencies
Before installing the package, install the dependencies specified in `requirements.txt` file.

# Installation
This package is also available on Pypi and thus one can directly pip install it.
```python
$ pip install cyclicgan
```
## Steps to build the API:
If you wanna contribute or build the package from repository, you can build the package as follows:
1. Clone the repo.
2. Run `python setup.py install`
3. API has been installed as a python package and now you can directly use its functionalities.

# Tutorial
You can follow [this](https://colab.research.google.com/drive/15vn4qxR66O_f_d67PxrO2yyHR2-5CczV#scrollTo=w4te2lF6CWib)
tutorial notebook to get an idea of the functionality of this API with a dataset.
