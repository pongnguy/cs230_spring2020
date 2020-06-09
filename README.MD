Authors: Shreyas, Rekha, Alfred
# Purpose
To train a fast question answering system on the Tensorflow 2.0 QA Challenge dataset, which contains user queries and labeled answers from a Wikipedia corpus.
This work was done as part of the CS230 Spring 2020 final project.


#Runnning the code

1. Download the Kaggle dataset from https://www.kaggle.com/c/tensorflow2-question-answering
2. Install Transformers 1.2.1, NVIDIA Apex 2.0 in a Python 3.6 virtualenv
3. Edit the location of training and validation files in train_qa.py and train_classifier.py
4. Train the question answering code.

python train_qa.py

5. Train the classifier. If you want to experiment with different hyperparameters, you can play around with the values. Each run will create a directory with a file named pytorch_model.bin, with a name like valid100_dr_0.1_unkWt_0.5lr2e-0507-06_07_50_47

./run.sh

7. Run the question answering system, using the directory created in the previous step. Make sure to use the same value for dropout as was used in the classifier. For example,

python3 -u eval.py --classifier_model_dir="valid100_dr_0.1_unkWt_0.5lr2e-0507-06_07_50_47"  --classifier_dropout=0.1 



