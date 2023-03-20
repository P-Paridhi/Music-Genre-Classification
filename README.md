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

### 1. LIBROSA
Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems. By using Librosa, we can extract certain key features from the audio samples such as Tempo, Chroma Energy Normalized, Mel-Freqency Cepstral Coefficients, Spectral Centroid, Spectral Contrast, Spectral Rolloff, and Zero Crossing Rate.

### 2. Python.display.Audio 
With the help of IPython.display.Audio we can play audio in the notebook.

### Visualizing Audio Files
Visualizing Audio Files: 
* Plot Raw Wave Files
* Spectrograms
* Spectral Rolloff
* Chroma Feature
* Zero Crossing Rate
