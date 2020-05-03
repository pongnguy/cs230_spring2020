"""
Main file for all the high-level functions/class

"""

#TODO
# 1) Instantiate a BERT model and feed in a (tokenized?) input string
# 2) TBD

import input as inputting
#import bert_qa.modeling as models
import json

json_dir = '../Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'

dataset_kaggle = inputting.input_datasets_Kaggle(json_dir, 10)
dataset_squad = inputting.format_KaggleToSquad(dataset_kaggle)

with open('squad_formatted.json', 'w') as f:
    f.seek(0)
    f.write(json.dumps(dataset_squad))

dataset_statistics = inputting.compute_statistics(dataset_kaggle)

print("Finished loading, formatting, and outputting the data")