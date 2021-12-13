# Overview
This project uses a Convolutional Neural Network (CNN) to classify images in the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset put together by the University of Toronto (Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton). 

**Technologies**
- Python
- Tensorflow (Keras)
- Numpy
- PIL Image
- Matplotlib
- Streamlit

# Data
- 32 x 32 (RGB)
- 50,000 Training Images
- 10,000 Testing Images

# Neural Network Architecture 
*Trainable Parameters: 122,570*
- Convolutional Layer 
- Max Pooling
- Convolutional Layer (x 2)
- Max Pooling (x 2)
- Flatten Layer 
- Dense Layer
- Dense Layer (x 2)