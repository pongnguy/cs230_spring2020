Authors: Shreyas, Rekha, Alfred
# Purpose
To train a fast question answering system on the Tensorflow 2.0 QA Challenge dataset, which contains user queries and labeled answers from a Wikipedia corpus.
This work was done as part of the CS230 Spring 2020 final project.


#Runnning the code

1. Download the Kaggle dataset from https://www.kaggle.com/c/tensorflow2-question-answering
2. Install Transformers 1.2.1, NVIDIA Apex 2.0 in a Python 3.6 virtualenv
3. Edit the location of training and validation files in train_qa.py and train_classifier.py
3.

python train_qa.py
python train_classifier.py
python eval.py

