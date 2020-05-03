"""
Main file for all the high-level functions/class

"""

#TODO
# 1) Instantiate a BERT model and feed in a (tokenized?) input string
# 2) TBD

import input as inputting

json_dir = '../Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'

dataset_kaggle = inputting.input_datasets_Kaggle(json_dir, 10)

print("Finished loading the data")