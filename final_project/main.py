"""
Main file for all the high-level functions/class

"""

#TODO
# 1) Instantiate a BERT model and feed in a (tokenized?) input string
# 2) TBD

#from Extend_BERT_as_QA_Chatbot import create_pretraining_data as ebqachat
import input as inputting
#import bert_qa.modeling as models
import json




json_dir = '../Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'

dataset_kaggle = inputting.input_datasets_Kaggle(json_dir, 1000)
dataset_squad = inputting.format_KaggleToSquad(dataset_kaggle)


with open('dataset_Kaggle_1000.json', 'w') as f:
    f.seek(0)
    f.write(json.dumps(dataset_kaggle))

#with open('squad_formatted.json', 'w') as f:
#    f.seek(0)
#    f.write(json.dumps(dataset_squad))

#dataset_statistics = inputting.compute_statistics(dataset_kaggle)




# Playing with creating a Tokenizer

#create_tokenizer_from_hub_module(bert_hub_module_handle):