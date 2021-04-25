# VAE-numpy
This project is a NumPy implementation of variational autoencoders from the paper "Auto-Encoding Variational Bayes" (Kingma and Welling, 2014), using the MNIST dataset for experimentation.

Take a look at [demo.ipynb](https://github.com/abhayran/VAE-numpy/blob/main/demo.ipynb) for a quick demonstration of results on MNIST, where the latent space distributions and output manifold are visualized:

Latent space distributions             |  Output space manifold
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/40629249/115994611-09bf8f00-a5d8-11eb-961b-2c6e1c53bf9a.png)  |  ![](https://user-images.githubusercontent.com/40629249/115994615-0a582580-a5d8-11eb-8879-5163f41d9ff3.png)

# Requirements
To experiment with this project, download the files train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz from [this link](http://yann.lecun.com/exdb/mnist), then extract their contents directly into the project directory. 

The [idx2numpy](https://pypi.org/project/idx2numpy/) library provides functionality for reading out the images directly into NumPy arrays. You can easily install this library with `pip install idx2numpy`
