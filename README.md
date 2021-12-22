## Gesture Classification for SmartHome

Requirements:<br>
* TensorFlow
* Python 3.6.9
* OpenCV for Python
* Keras

Usage:<br>
* Training data folder: 'traindata'
* Make sure that the naming convention includes keywords for the gesture.
* Test data folder: 'test'
* Run```python main.py```
* A Results.csv will be generated in the same directory with as the project which will have the label of the predicted gestures.

Description: <br>
- cnn_model.h5 is used which is a pretrained model on the hand alphabets which is
close to the hand gestures that we are trying to recognize.
- Used frameextractor.py which has functionality to extract the middle frame
from the sequence of all the frames in the videos. It resizes the image in a
square and has other utility functions for making image processing.
- Used handshape_feature_extractor.py which we use along with the
cnn_model for extracting the feature vector or the penultimate layer of the CNN.
- We use our training data and find out the feature vectors for 17 gestures and
store them in a file training.csv.
- For the feature vector extraction we make use of the frameextractor.py to get .png
files in a new folder named as ‘extractedFrames’
- For the test phase, for each video, we again find out the image of the middle frame
and store it in a folder named ‘extractedFramesTest’. After the .png files are extracted, we find out the feature vector of the image and find the cosine similarity of it with the 17 gesture vectors so that we get the most similar gesture.
- Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space.
- Thus we predict the gesture of the test video and recognize its gesture
