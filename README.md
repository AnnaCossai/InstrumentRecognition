# 🎹🎸🎺 Musical Instrument Recognition

This repository contains the code developed for the thesis for the BSc in Computer and Systems Engineering in Sapienza University of Rome.


## 🚀 Getting Started

The code relies on the Python libraries [librosa](https://librosa.org/doc/latest/index.html) and [PyTorch](https://pytorch.org/get-started/locally/) 


## 🎵 Prepare the data
Download the Medley-solo-DB dataset from [here](https://zenodo.org/records/3464194). Place the audio files in a folder called `8_instruments_dataset`.

## 🎼 Generate mel-spectrograms

To generate mel-spectrograms from the audio files, run the script ``MelSpectrogramGeneration.py``

## 🏋️‍♂️🤖 Train and test the models

To train and test the CNN 1D on raw audio file, run the script ``InstrumentRecognitionFromRawAudio.py``.

To train and test the CNN 2D on mel-spectrograms, run the script ``InstrumentRecognition_CNN_MelSpectrogram.py``
