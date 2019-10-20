# mnist_domain_adaptation

Suppose you wish to train a net to classify some real images, but only have labeled data for synthesized images. For example, you might want to train a net to classify handwritten digits, but your only labeled data consists of synthesized images. If you train a classifier on the synthesized data, then it will perform poorly on the real data.

To solve this, you can add an "adversial" adapter to the start of the net. Specifically, you train three nets at once: an adapter, a discriminator, and a classifier. You input both the real and synthesized images into the adapter and input the adapter output into the discriminator. You train the discriminator to distinguish between the adapter outputs of real and synthesized images, and you train the adapter to convert both the real and synthesized images into the same subset of output space. This is similar to a GAN, except instead of generating fake images to look like real images you're modifying real images to make them look like synthesized images.

This model performs precisely this domain adaptation in the case of synthesized digits (from the Chars74k dataset) and handwritten digits (from the MNIST dataset). Of course, this model is entirely academic: the MNIST dataset has labels and it would be best to train it with its own labels.