"""
Main file for all the high-level functions/class

"""

#TODO
# 1) Instantiate a BERT model and feed in a (tokenized?) input string
# 2) TBD


from .preprocess import jsonlToJson, format_KaggleToSquad
#import bert_qa.modeling as models
import json


# Format the


# Translate JSONL to JSON
#dataset_simplified = inputting.jsonlToJson('../Guanshuo_TFQA_1stplace/input/natural_questions/simplified-nq-valid.jsonl')


json_dir = '../Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'
#json_dir = '../Guanshuo_TFQA_1stplace/input/simplified-nq-train.jsonl'

num_entries = 1000

dataset_kaggle = jsonlToJson(json_dir, num_entries)
dataset_squad = format_KaggleToSquad(dataset_kaggle)

#dataset_kaggle_statistics = inputting.compute_statistics(dataset_kaggle)



with open('squad_formatted_' + str(num_entries) + '.json', 'w') as f:
    f.seek(0)
    f.write(json.dumps(dataset_squad))



#with open('../Guanshuo_TFQA_1stplace/input/natural_questions/simplified-nq-valid.json', 'w') as f:
#    f.seek(0)
#    #output = json.dumps(dataset_simplified, ensure_ascii=False).encode('utf16')
#    f.write(json.dumps(dataset_squad, ensure_ascii=False))

#with open('squad_formatted.json', 'w') as f:
#    f.seek(0)
#    f.write(json.dumps(dataset_squad))

#dataset_statistics = inputting.compute_statistics(dataset_kaggle)




# Playing with creating a Tokenizer

#create_tokenizer_from_hub_module(bert_hub_module_handle):