# Cat-Face-Generation-using-Generative-Adversarial-Network-
This repository contains code for training a Generative Adversarial Network (GAN) to generate high-quality cat face images. The GAN is trained on a dataset of approximately 20,000 cat face images.

Dataset
The cat images used for training are RGB images with a resolution of 64x64 pixels. To prevent the discriminator from overfitting, data augmentation techniques are applied during training. These techniques include random cropping, color jittering, and random adjustment of sharpness.

Model Architecture
The GAN consists of a Generator and a Discriminator. The Generator takes random noise as input and generates fake cat face images. The Discriminator is trained to distinguish between real and fake cat face images.

The Discriminator architecture consists of several convolutional layers with leaky ReLU activation and batch normalization. The final layer is a convolutional layer that outputs a single scalar value indicating the likelihood of the input being real or fake.

The Generator architecture is a mirror image of the Discriminator, with transposed convolutional layers and ReLU activation. The final layer uses the hyperbolic tangent (tanh) activation to generate pixel values in the range [-1, 1].

Training
The GAN is trained using the Least-Squares GAN (LSGAN) loss function. The discriminator loss is computed using the mean squared error (MSE) loss, while the generator loss is computed using the binary cross-entropy loss.

During training, the discriminator and generator are updated alternately. The discriminator is trained to minimize the discriminator loss, while the generator is trained to minimize the generator loss.

Usage
To train the GAN, run the train function, providing the appropriate arguments such as the discriminator and generator models, optimizers, loss functions, batch size, number of epochs, and the training dataset. The training progress and generated images will be displayed at regular intervals.

Requirements
PyTorch
TorchVision
Matplotlib
Acknowledgments
The implementation of the GAN and the training process was inspired by the original GAN paper by Ian Goodfellow et al. (https://arxiv.org/abs/1406.2661).

The code in this repository can serve as a starting point for training GANs on other image datasets and generating synthetic images.

![image](https://user-images.githubusercontent.com/98642342/235659000-dac4f84a-36d2-451d-be66-7d62a32aa28a.png)
