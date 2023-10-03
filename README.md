# CNN for CIFAR-10 using TF2.0
This is some implementations of CNN Architecture for CIFAR-10 dataset using Tensorflow and Keras Functional API. The dataset consists of airplanes, dogs, cats, and other objects. We first did preprocessing on the images, then trained a convolutional neural network on all the samples. 

![cifar10][1]

# 2. Understanding the dataset
The original a batch data is (10000 x 3072) dimensional tensor expressed in numpy array, where the number of columns, (10000), indicates the number of sample data. As stated in the [CIFAR-10/CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), the row vector, (3072) represents an color image of 32x32 pixels.

Since this project is going to use CNN for the classification tasks, the row vector, (3072), is not an appropriate form of image data to feed. In order to feed an image data into a CNN model, the dimension of the tensor representing an image data should be either (width x height x num_channel) or (num_channel x width x height).

It depends on your choice (check out the [tensorflow conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)). In this particular project, I am going to use the dimension of the first choice because the default choice in tensorflow's CNN operation is so.

The row vector (3072) has the exact same number of elements if you calculate 32\*32\*3==3072. In order to reshape the row vector, (3072), there are two steps required. The **first** step is involved with using **reshape** function in numpy, and the **second** step is involved with using **transpose** function in numpy as well.

![reshape-transpose.png][3]

# 5. Model Architecture

![con_model.png][4]

at first we used strided convolutional layers which resulted in low accuracy of the model. That's why to get better results in the improved version we used Batch Normalization, taking inspiration from VGG networks, by doing multiple strided convolution layers before doing pooling. So the improved model consists of 15 layers in total. In addition to layers below lists what techniques are applied to build the model.

1. Convolution with 32 different filters in size of (3x3)
2. Batch Normalization
3. Convolution with 32 different filters in size of (3x3)
4. Batch Normalization
5. Max Pooling by 2
6. Convolution with 64 different filters in size of (3x3)
7. Batch Normalization
8. Convolution with 64 different filters in size of (3x3)
9. Batch Normalization
10. Max Pooling by 2
11. Convolution with 128 different filters in size of (3x3)
12. Batch Normalization
13. Convolution with 128 different filters in size of (3x3)
14. Batch Normalization
15. Max Pooling by 2
Then we flatten out the 3D layers and add some dropout layers to reduce overfitting.

the image below decribes how the conceptual convolving operation differs from the tensorflow implementation when you use [Channel x Width x Height] tensor format.

![convolving][5]

# 6. Training the model
After doing improving the model, we achieve over 87% accuracy in 50 epochs.

![training][6]

To take a closer look on how the improved model helped us reduce overfitting we can look on accuracy graphs for before & after changing the model by implementing batch normalization and data augmentation.

![Accuracy Before][7]

![Accuracy after][8]





  [1]: ./media/cf10.png
  [2]: ./media/cf10.png
  [3]: ./media/reshape-transpose.png
  [4]: ./media/conv_model.png
  [5]: ./media/convolving.png
  [6]: ./media/training_improved.jpg
  [7]: ./media/acc.jpg
  [8]: ./media/acc_improved.jpg
