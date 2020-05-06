"""
Main file for all the high-level functions/class

"""

#TODO
# 1) Instantiate a BERT model and feed in a (tokenized?) input string
# 2) TBD

#from Extend_BERT_as_QA_Chatbot import create_pretraining_data as ebqachat
import preprocess as inputting
#import bert_qa.modeling as models
import json




json_dir = '../Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'
#json_dir = '../cs230_spring2020/Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'

num_entries = 10

dataset_kaggle = inputting.input_datasets_Kaggle(json_dir, num_entries)
dataset_squad = inputting.format_KaggleToSquad(dataset_kaggle)

dataset_kaggle_statistics = inputting.compute_statistics(dataset_kaggle)

with open('dataset_Kaggle_' + str(num_entries) + '.json', 'w') as f:
    f.seek(0)
    f.write(json.dumps(dataset_kaggle))

#with open('squad_formatted.json', 'w') as f:
#    f.seek(0)
#    f.write(json.dumps(dataset_squad))

#dataset_statistics = inputting.compute_statistics(dataset_kaggle)




# Playing with creating a Tokenizer

#create_tokenizer_from_hub_module(bert_hub_module_handle):