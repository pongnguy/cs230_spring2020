




import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import jsonlines
from test_utils import simplify_nq_example
import json


json_dir = 'v1.0-simplified_nq-dev-all.jsonl'
dict_list = []

try:
    f = open(file=json_dir, encoding='utf-8')
except:
    f = open(file=json_dir, encoding='utf-16')

# with open(file=json_dir, encoding='utf-16') as f:
for line in tqdm(f):
    dict_list.append(simplify_nq_example(json.loads(line)))

with jsonlines.open('simplified-nq-valid.jsonl', 'w') as writer:
    writer.write_all(dict_list)