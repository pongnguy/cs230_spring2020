"""
General file for inputting the data

"""

from tqdm import tqdm
import json
import numpy as np
from typing import List, TypedDict


DatasetKaggle = TypedDict('DatasetKaggle', {'document_text': str, 'long_answer_candidates': List[TypedDict('long_answer_candidate', {'start_token': int, 'top_level': bool, 'end_token': int})], 'question_text': str, 'annotations': List[TypedDict('annotation', {'yes_no_answer': str, 'long_answer': TypedDict('long_answer', {'start_token': int, 'candidate_index': int, 'end_token': int}), 'short_answers': List[TypedDict('short_answer', {'start_token': int, 'end_token': int})], 'annotation_id': int})], 'document_url': str, 'example_id': int})



def random_sample_negative_candidates(distribution):
    temp = np.random.random()
    value = 0.
    for index in range(len(distribution)):
        value += distribution[index]
        if value > temp:
            break
    return index

def input_datasets_Kaggle(json_dir, max_data = 9999999999):
    """
    Inputting the data from Kaggle

    :param json_dir:  path to the filename
    :type json_dir:  str
    :param max_data:  maximum number of rows to input
    :type max_data:  int
    :returns: processed dataset
    :rtype: dict
    """

    # prepare input
    #json_dir = '../../input/simplified-nq-train.jsonl'
    #max_data = 9999999999

    id_list = []
    data_dict = {}
    with open(json_dir) as f:
        for n, line in tqdm(enumerate(f)):
            if n > max_data:
                break
            data = json.loads(line)

            is_pos = False
            annotations = data['annotations'][0]
            if annotations['yes_no_answer'] == 'YES':
                is_pos = True
            elif annotations['yes_no_answer'] == 'NO':
                is_pos = True
            elif annotations['short_answers']:
                is_pos = True
            elif annotations['long_answer']['candidate_index'] != -1:
                is_pos = True

            if is_pos and len(data['long_answer_candidates'])>1:
                data_id = data['example_id']
                id_list.append(data_id)

                # uniform sampling
                distribution = np.ones((len(data['long_answer_candidates']),),dtype=np.float32)
                if is_pos:
                    distribution[data['annotations'][0]['long_answer']['candidate_index']] = 0.
                distribution /= len(distribution)
                negative_candidate_index = random_sample_negative_candidates(distribution)

                #
                doc_words = data['document_text'].split()
                # negative
                candidate = data['long_answer_candidates'][negative_candidate_index]
                negative_candidate_words = doc_words[candidate['start_token']:candidate['end_token']]
                negative_candidate_start = candidate['start_token']
                negative_candidate_end = candidate['end_token']
                # positive
                candidate = data['long_answer_candidates'][annotations['long_answer']['candidate_index']]
                positive_candidate_words = doc_words[candidate['start_token']:candidate['end_token']]
                positive_candidate_start = candidate['start_token']
                positive_candidate_end = candidate['end_token']

                # initialize data_dict
                data_dict[data_id] = {
                                      'question_text': data['question_text'],
                                      'annotations': data['annotations'],
                                      'positive_text': positive_candidate_words,
                                      'positive_start': positive_candidate_start,
                                      'positive_end': positive_candidate_end,
                                      'negative_text': negative_candidate_words,
                                      'negative_start': negative_candidate_start,
                                      'negative_end': negative_candidate_end,
                                     }