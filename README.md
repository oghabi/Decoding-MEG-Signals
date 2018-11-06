
## Decoding Brain MEG Signals Using Deep Neural Networks

This is a [Kaggle competition](https://www.kaggle.com/c/decoding-the-human-brain).  Subjects are shown two different images and their brain MEG signals are recorded at each moment for 1.5 seconds. The goal is to use the MEG signals in order to classify which image the subject was looking at during the respective session. 

Deep learning models are implemented to decoded brain MEG signals to classify the visual stimuli shown to the subjects.


## Dataset
The dataset used is on [Kaggle](https://www.kaggle.com/c/decoding-the-human-brain).  

## Results
[Report](https://github.com/oghabi/Decoding-MEG-Signals/blob/master/Report.pdf) is a brief report I wrote outlining the pre-processing done on the dataset and the various deep models that were implemented along with the evaluation of each model on the Kaggle test set.

Most of the contestants in this competition used some form of feature engineering along with traditional models (such as a combination of logistic regression and random forest. 

I attempted to tackle this problem using deep learning without the use of feature engineering. My model resulted in my score being anked above 90th percentile in Kaggle leaderboard.


