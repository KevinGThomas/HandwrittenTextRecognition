# HandwrittenTextRecognition
Recognizing Handwritten Text by segmenting the page into paragraphs and lines and then converting them to digital text.

# Overview
This is the full code for 'Handwritten Text Recognition'. This code helps to convert a handwritten page into digital text by identifying the paragraph present in the page, segmenting the lines and running word recognition to accurately identify the text.

# Dependency
* mxnet
* pandas
* matplotlib
* numpy
* skimage

# Methodology
## Paragraph Segmentation

<img align="right" width="110" src="https://user-images.githubusercontent.com/20180559/67067618-a1971e00-f194-11e9-896a-4940092e26ec.png">

### Pre-processing
The pre-processing is a series of operations performed of scanned input image. It essentially enhances the image rendering for suitable segmentation. The role of pre-processing is to segment the interesting pattern from the background. Methods like data augmentation (a copy of the input image is made and slight alterations such as small rotation of the image are done and both of these images are sent to the model to increase it’s dataset examples), grey-scaling (images are turned to black and white for the model to accurately detect the presence of handwritten text).

### Feature Extraction
The art of finding useful features in a machine learning problem can be tedious and heavily affected by human bias but by using Convolution Neural Networks, we are able to detect features by itself by comparing similar patterns in the images. To extract features from the DCNN model, first we need to train the CNN network with the last sigmoid/logistic dense layer (here dimension 4). The objective of the training network is to identify the correct weight for the network by multiple forward and backward iterations, which eventually try to minimise mean square error. We use MXNet in order to solve the problem for a given problem and set MSE (Mean Square Error) as the evaluation metric. We will optimize the model by attempting to reduce MSE value in each new epoch.

### Segmentation
Here is the architecture of the DCNN model.

The model gives 4 values as output in the end, (x,y,w,h). (x,y) are the coordinates of the starting of the paragraph that the model has recognized, w is the width of the paragraph and h is the height of the paragraph. Using this parameters, a bounding box can be formed around the paragraph to successfully segment the paragraph from the given image.

## Line Segmentation
Similarly, line segmentation is done through pre-processing, feature extraction and segmentation. Line Segmentation is used to identify the lines present in the paragraph. This is important as many people have a tendency to not write in a straight line.
<img src="https://user-images.githubusercontent.com/20180559/67068121-9e9d2d00-f196-11e9-945f-3ff896e8fd51.png">

Here is the architecture of the SSD network model.

The model contains a list of bounding boxes each containing 4 values as output in the end, [n][(x,y,w,h)]. n is the number of words detected in the paragraph, (x,y) are the coordinates of the starting of the word that the model has recognized, w is the width of the word and h is the height of the word. Using this parameters, a bounding box can be formed around each word to successfully detect the words from the given image to segment to lines (checks if y coordinate of the bounding boxes overlap each other).

## Handwriting Recognition
The final model is the handwriting recognition model which takes a line as input and converts the line into digital text. This model consits of a CNN-biLSTM architecture. The loss used is the CTC (Connectionist Temporal Classification) loss. 
<p align="center">
<img align="center" height="400" src="https://user-images.githubusercontent.com/20180559/67068512-ea040b00-f197-11e9-8665-8afa5daf00f6.png">
</p>
 
Here is the CNN-biLSTM architecture model.

The input lines are sent into the CNN to extract features from similar patterns. These image features are then sent to a sequential learner which are the  bidirectional LSTMs which are then sent to the output string that predict the character based on the alphabet with the highest predicted value given by the model.

# Results
## Paragraph Segmentation
<img src="https://user-images.githubusercontent.com/20180559/67069088-8ed31800-f199-11e9-9ff1-ce93c7a59143.jpg">

## Line Segmentation
<img src="https://user-images.githubusercontent.com/20180559/67069187-eec9be80-f199-11e9-8338-f6254e27afda.jpg">

## Handwriting Recognition
<img src="https://user-images.githubusercontent.com/20180559/67069304-449e6680-f19a-11e9-9846-c25ba51c2a7c.jpg">
