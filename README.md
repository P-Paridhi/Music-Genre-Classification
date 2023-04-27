# Music-Genre-Classification

## Dataset
For this project, the dataset that we will be working with is GTZAN Genre Classification dataset which consists of 1,000 audio tracks, each 30 seconds long. It contains 10 genres, each represented by 100 tracks.
The 10 genres are as follows:

* Blues

* Classical

* Country

* Disco

* Hip-hop

* Jazz

* Metal

* Pop

* Reggae

* Rock

The dataset has the following folders:

Genres original — A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)

Images original — A visual representation for each audio file. One way to classify data is through neural networks because NN’s usually take in some sort of image representation.

2 CSV files — Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs are split before into 3 seconds audio files.

## Audio Libraries Used

### LIBROSA
Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems. By using Librosa, we can extract certain key features from the audio samples such as Tempo, Chroma Energy Normalized, Mel-Freqency Cepstral Coefficients, Spectral Centroid, Spectral Contrast, Spectral Rolloff, and Zero Crossing Rate.


### Feature Extraction 
- Load the data and extract the MFCC features
- if mfcc shape is less than input shape then padding is performed on mfcc with '0'.

### Dividing Data Into Training and Testing Sets
- 80% training, 20% testing 

### Building a Model 
- There are many ways through which we can train our model. Some of these approaches are:
  - Multiclass Support Vector Machines
  - K-Means Clustering
  - K-Nearest Neighbors
  - Convolutional Neural Networks
  ## The chart shows why we are using CNN to train our model: 
  ![image](https://user-images.githubusercontent.com/109361931/229408289-430c83fe-0996-4287-8f50-bb0de5a95bf9.png)

- Optimizer used: Adam 
- learning rate: 0.001 
- loss: categorical_crossentropy 
- No. of epoch: 100
- All of the hidden layers are using the RELU activation function and the output layer uses the softmax function.
- Dropout is used to prevent overfitting.      

Evaluation 
- Test loss: 0.28
- Test Accuracy: 90%
