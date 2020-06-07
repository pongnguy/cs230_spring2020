import argparse
from collections import defaultdict
from dataclasses import dataclass
import functools
import gc
import itertools
import json
from multiprocessing import Pool
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from typing import Callable, Dict, List, Generator, Tuple

import numpy as np
import pandas as pd
from pandas.io.json._json import JsonReader
from sklearn.preprocessing import LabelEncoder
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader

from apex import amp
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel, DistilBertModel, DistilBertConfig

from models import DistilBertForTFQA

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='/data/global_data/rekha_data', help="directory containing training and validation files")
parser.add_argument("--classifier_model_dir", type=str, default='/data/sv/distilbert', help="model file")
parser.add_argument("--classifier_dropout", type=float, default=0.1, help="-")
parser.add_argument("--qa_dropout", type=float, default=0.1, help="-")
parser.add_argument("--valid_size", type=int, default=100, help="number of validation examples")
parser.add_argument("--batch_size", type=int, default=16, help="batch size to use for training examples")
parser.add_argument("--fp16", type=bool, default=False, help="whether to use 16-bit precision")
parser.add_argument("--hidden_layers", type=int, default=6, help="number of hidden layers from pretrained model to use")

args = parser.parse_args()

VALID_SIZE = args.valid_size
#EVAL_DATA_PATH = '/data/global_data/rekha_data/simplified-nq-valid_'+str(VALID_SIZE)+'_set2.jsonl'
EVAL_DATA_PATH = '/data/global_data/rekha_data/train_head6k_tail100_total100.jsonl'
chunksize = 1000

#
# get_ipython().system('wc -l $DATA_PATH')
# get_ipython().system('wc -l $EVAL_DATA_PATH')
DEBUG = True

# In[ ]:


# DATA_DIR = Path('../input/tensorflow2-question-answering/')
# DATA_PATH = DATA_DIR / 'simplified-nq-train.jsonl'

start_time = time.time()

seed = 1029
valid_size = VALID_SIZE

max_seq_len = 384
max_question_len = 64
doc_stride = 128

num_labels = 5
n_epochs = 1
lr = 2e-5
warmup = 0.05
batch_size = 16
accumulation_steps = 4
DROPOUT=0.2

qa_model_name = 'bert-base-uncased'
classifier_model_name = 'distilbert-base-uncased-distilled-squad'

do_lower_case = 'uncased' in qa_model_name
device = torch.device('cuda')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# In[ ]:


@dataclass
class Example(object):
    example_id: int
    candidates: List[Dict]
    annotations: Dict
    doc_tokens: List[str]
    doc_start: int
    question_len: int
    tokenized_to_original_index: List[int]
    input_ids: List[int]
    start_position: int
    end_position: int
    class_label: str


def convert_data(
        line: str,
        tokenizer: BertTokenizer,
        max_seq_len: int,
        max_question_len: int,
        doc_stride: int
) -> List[Example]:
    """Convert dictionary data into list of training data.

    Parameters
    ----------
    line : str
        Training data.
    tokenizer : transformers.BertTokenizer
        Tokenizer for encoding texts into ids.
    max_seq_len : int
        Maximum input sequence length.
    max_question_len : int
        Maximum input question length.
    doc_stride : int
        When splitting up a long document into chunks, how much stride to take between chunks.
    """

    def _find_short_range(short_answers: List[Dict]) -> Tuple[int, int]:
        answers = pd.DataFrame(short_answers)
        start_min = answers['start_token'].min()
        end_max = answers['end_token'].max()
        return start_min, end_max

    # model input
    data = json.loads(line)
    doc_words = data['document_text'].split()
    question_tokens = tokenizer.tokenize(data['question_text'])[:max_question_len]

    # tokenized index of i-th original token corresponds to original_to_tokenized_index[i]
    # if a token in original text is removed, its tokenized index indicates next token
    original_to_tokenized_index = []
    tokenized_to_original_index = []
    all_doc_tokens = []  # tokenized document text
    for i, word in enumerate(doc_words):
        original_to_tokenized_index.append(len(all_doc_tokens))
        if re.match(r'<.+>', word):  # remove paragraph tag
            continue
        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            tokenized_to_original_index.append(i)
            all_doc_tokens.append(sub_token)

    # model output: (class_label, start_position, end_position)
    annotations = data['annotations'][0]
    if annotations['yes_no_answer'] in ['YES', 'NO']:
        class_label = annotations['yes_no_answer'].lower()
        start_position = annotations['long_answer']['start_token']
        end_position = annotations['long_answer']['end_token']
    elif annotations['short_answers']:
        class_label = 'short'
        start_position, end_position = _find_short_range(annotations['short_answers'])
    elif annotations['long_answer']['candidate_index'] != -1:
        class_label = 'long'
        start_position = annotations['long_answer']['start_token']
        end_position = annotations['long_answer']['end_token']
    else:
        class_label = 'unknown'
        start_position = -1
        end_position = -1

    # convert into tokenized index
    if start_position != -1 and end_position != -1:
        start_position = original_to_tokenized_index[start_position]
        end_position = original_to_tokenized_index[end_position]

    # make sure at least one object in `examples`
    examples = []
    max_doc_len = max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]

    # take chunks with a stride of `doc_stride`
    for doc_start in range(0, len(all_doc_tokens), doc_stride):
        doc_end = doc_start + max_doc_len
        # if truncated document does not contain annotated range
        if not (doc_start <= start_position and end_position <= doc_end):
            start, end, label = -1, -1, 'unknown'
        else:
            start = start_position - doc_start + len(question_tokens) + 2
            end = end_position - doc_start + len(question_tokens) + 2
            label = class_label

        assert -1 <= start < max_seq_len, f'start position is out of range: {start}'
        assert -1 <= end < max_seq_len, f'end position is out of range: {end}'

        doc_tokens = all_doc_tokens[doc_start:doc_end]
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        examples.append(
            Example(
                example_id=data['example_id'],
                candidates=data['long_answer_candidates'],
                annotations=annotations,
                doc_tokens=doc_words,
                doc_start=doc_start,
                question_len=len(question_tokens),
                tokenized_to_original_index=tokenized_to_original_index,
                input_ids=tokenizer.convert_tokens_to_ids(input_tokens),
                start_position=start,
                end_position=end,
                class_label=label
            ))

    return examples


# In[ ]:


class JsonChunkReader(JsonReader):
    """JsonReader provides an interface for reading in a JSON file.
    """

    def __init__(
            self,
            filepath_or_buffer: str,
            convert_data: Callable[[str], List[Example]],
            orient: str = None,
            typ: str = 'frame',
            dtype: bool = None,
            convert_axes: bool = None,
            convert_dates: bool = True,
            keep_default_dates: bool = True,
            numpy: bool = False,
            precise_float: bool = False,
            date_unit: str = None,
            encoding: str = None,
            lines: bool = True,
            chunksize: int = 2000,
            compression: str = None,
    ):
        super(JsonChunkReader, self).__init__(
            str(filepath_or_buffer),
            orient=orient, typ=typ, dtype=dtype,
            convert_axes=convert_axes,
            convert_dates=convert_dates,
            keep_default_dates=keep_default_dates,
            numpy=numpy, precise_float=precise_float,
            date_unit=date_unit, encoding=encoding,
            lines=lines, chunksize=chunksize,
            compression=compression
        )
        self.convert_data = convert_data

    def __next__(self):
        lines = list(itertools.islice(self.data, self.chunksize))
        # for line in lines:
        # print(line)
        # print('Length of lines',len(lines), 'chunksize', self.chunksize)
        if lines:
            with Pool(2) as p:
                obj = p.map(self.convert_data, lines)
                # print('Length of obj', len(obj))
            return obj

        self.close()
        raise StopIteration



class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for QA and classification tasks.

    Parameters
    ----------
    config : transformers.BertConfig. Configuration class for BERT.

    Returns
    -------
    start_logits : torch.Tensor with shape (batch_size, sequence_size).
        Starting scores of each tokens.
    end_logits : torch.Tensor with shape (batch_size, sequence_size).
        Ending scores of each tokens.
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Classification scores of each labels.
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # predict start & end position
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        return start_logits, end_logits, classifier_logits

# In[ ]:


def loss_fn(preds, labels):
    start_preds, end_preds, class_preds = preds
    start_labels, end_labels, class_labels = labels

    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    class_loss = nn.CrossEntropyLoss()(class_preds, class_labels)
    return start_loss + end_loss + class_loss


def loss_fn_classifier(preds, labels):
    _,_, class_preds = preds
    _, _,class_labels = labels

    class_loss = nn.CrossEntropyLoss()(class_preds, class_labels)

    return class_loss
tokenizer = BertTokenizer.from_pretrained(qa_model_name, do_lower_case=True)
convert_func = functools.partial(convert_data,
                                 tokenizer=tokenizer,
                                 max_seq_len=max_seq_len,
                                 max_question_len=max_question_len,
                                 doc_stride=doc_stride)

config = BertConfig.from_pretrained(qa_model_name)
config.num_labels = 5
qa_model = BertForQuestionAnswering.from_pretrained('/data/sv/bert_trained', config=config)

config = DistilBertConfig.from_pretrained(classifier_model_name)
config.num_labels = 5
config.dropout = args.classifier_dropout
classifier_model = DistilBertForTFQA.from_pretrained(args.classifier_model_dir, config=config,
                                                     hidden_layers=args.hidden_layers, batch_size=args.batch_size)

def eval_collate_fn(examples: List[Example]) -> Tuple[List[torch.Tensor], List[Example]]:
    # input tokens
    max_len = max([len(example.input_ids) for example in examples])
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids)]

    return inputs, examples

class ExampleBatch:
    def __init__(self):
        self.inputs = []
        self.examples = []

    def update(self,
               inputs: List[torch.Tensor],
               examples: List[Example]):
        inputs = [element.unsqueeze(0) if element.dim() < 2 else element for element in inputs]
        if not self.inputs:
            self.inputs = inputs
        else:
            for i,element in enumerate(self.inputs):
                self.inputs[i] = torch.cat((element, inputs[i]), 0)

        self.examples = self.examples + examples
    def clear(self):
        self.inputs.clear()
        self.examples.clear()

def eval_model(
        qa_model: nn.Module,
        classifier_model: nn.Module,
        valid_loader: DataLoader,
        device: torch.device = torch.device('cuda')
) -> Dict[str, float]:
    """Compute validation score.

    Parameters
    ----------
    model : nn.Module
        Model for prediction.
    valid_loader : DataLoader
        Data loader of validation data.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    dict
        Scores of validation data.
        `long_score`: score of long answers
        `short_score`: score of short answers
        `overall_score`: score of the competition metric
    """
    qa_model.to(device)
    if args.fp16:
        qa_model.half()
    qa_model.eval()
    classifier_model.to(device)
    classifier_model.eval()
    class_labels = ['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']
    unknown_label = class_labels.index('UNKNOWN')
    with torch.no_grad():
        result = Result()
        classifier_rejects = ExampleBatch()
        classifier_forwards = ExampleBatch()
        for inputs, examples in tqdm(valid_loader):
            input_ids, attention_mask, token_type_ids = inputs
            y_preds1 = classifier_model(input_ids.to(device),
                             attention_mask.to(device))
            _,_,classifier_preds = (p.detach().cpu() for p in y_preds1)

            has_pred = (torch.argmax(classifier_preds, dim=1)) != unknown_label
            print(has_pred)
            neg_pred = ~has_pred
            qa_inputs = [element[(has_pred != 0).nonzero().squeeze()] for element in inputs]
            qa_examples = (np.array(examples)[has_pred.numpy()]).tolist()
            if qa_examples:
                classifier_forwards.update(qa_inputs, qa_examples)
            reject_inputs = [element[(neg_pred != 0).nonzero().squeeze()]  for element in inputs]
            reject_examples = (np.array(examples)[neg_pred.numpy()]).tolist()
            if reject_examples:
                classifier_rejects.update(reject_inputs, reject_examples)

            if len(classifier_forwards.examples) >= batch_size:
                qa_inputs = classifier_forwards.inputs
                qa_examples = classifier_forwards.examples
                qa_input_ids, qa_attention_mask, qa_token_type_ids = qa_inputs

                y_preds = qa_model(qa_input_ids.to(device),
                                qa_attention_mask.to(device),
                                qa_token_type_ids.to(device))

                start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds)
                start_logits, start_index = torch.max(start_preds, dim=1)
                end_logits, end_index = torch.max(end_preds, dim=1)

                # span logits minus the cls logits seems to be close to the best
                cls_logits = start_preds[:, 0] + end_preds[:, 0]  # '[CLS]' logits
                logits = start_logits + end_logits - cls_logits  # (batch_size,)
                indices = torch.stack((start_index, end_index)).transpose(0, 1)  # (batch_size, 2)
                result.update(qa_examples, logits.numpy(), indices.numpy(), class_preds.numpy())
                print("Passed "+str(len(classifier_forwards.examples))+" to stage 2")
                classifier_forwards.clear()
            if len(classifier_rejects.examples) >= batch_size:
                reject_examples = classifier_rejects.examples
                start_index = torch.full([len(classifier_rejects.examples)], -1)
                end_index = torch.full([len(classifier_rejects.examples)], -1)
                indices = torch.stack((start_index, end_index)).transpose(1, 0)
                result.update(reject_examples, np.zeros(len(classifier_rejects.examples)), indices, np.zeros(len(classifier_rejects.examples)))
                print("Rejected "+str(len(classifier_rejects.examples))+" in classifier iteself")
                classifier_rejects.clear()
        if classifier_forwards.examples:
            qa_inputs = classifier_forwards.inputs
            qa_examples = classifier_forwards.examples
            qa_input_ids, qa_attention_mask, qa_token_type_ids = qa_inputs

            y_preds = qa_model(qa_input_ids.to(device),
                               qa_attention_mask.to(device),
                               qa_token_type_ids.to(device))

            start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds)
            start_logits, start_index = torch.max(start_preds, dim=1)
            end_logits, end_index = torch.max(end_preds, dim=1)

            # span logits minus the cls logits seems to be close to the best
            cls_logits = start_preds[:, 0] + end_preds[:, 0]  # '[CLS]' logits
            logits = start_logits + end_logits - cls_logits  # (batch_size,)
            indices = torch.stack((start_index, end_index)).transpose(1, 0)  # (batch_size, 2)
            result.update(qa_examples, logits.numpy(), indices.numpy(), class_preds.numpy())
            print("Passed " + str(len(classifier_forwards.examples)) + " to stage 2 at the end")
        if classifier_rejects.examples:
            start_index = torch.full([len(classifier_rejects.examples)], -1)
            end_index = torch.full([len(classifier_rejects.examples)], -1)
            indices = torch.stack((start_index, end_index)).transpose(1, 0)
            result.update(classifier_rejects.examples, np.zeros(len(classifier_rejects.examples)),
                          indices, np.zeros(len(classifier_rejects.examples)))
            print("Rejected " + str(len(classifier_rejects.examples)) + " in classifier itself at the end")
    return result.score()


class Result(object):
    """Stores results of all test data.
    """

    def __init__(self):
        self.examples = {}
        self.results = {}
        self.best_scores = defaultdict(float)
        self.class_labels = ['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']

    @staticmethod
    def is_valid_index(example: Example, index: List[int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if start_index > end_index:
            return False
        if start_index <= example.question_len + 2:
            return False
        return True

    def update(
            self,
            examples: List[Example],
            logits: torch.Tensor,
            indices: torch.Tensor,
            class_preds: torch.Tensor
    ):
        """Update batch objects.

        Parameters
        ----------
        examples : list of Example
        logits : np.ndarray with shape (batch_size,)
            Scores of each examples..
        indices : np.ndarray with shape (batch_size, 2)
            `start_index` and `end_index` pairs of each examples.
        class_preds : np.ndarray with shape (batch_size, num_classes)
            Class predicition scores of each examples.
        """
        for i, example in enumerate(examples):
            if self.is_valid_index(example, indices[i]) and self.best_scores[example.example_id] <= logits[i]:
                self.best_scores[example.example_id] = logits[i]
                self.examples[example.example_id] = example
                self.results[example.example_id] = [
                    example.doc_start, indices[i], class_preds[i]]
                print("update: now results len is:"+str(len(self.results.keys())))

    def _generate_predictions(self) -> Generator[Dict, None, None]:
        """Generate predictions of each examples.
        """
        for example_id in self.results.keys():
            doc_start, index, class_pred = self.results[example_id]
            example = self.examples[example_id]
            tokenized_to_original_index = example.tokenized_to_original_index
            if doc_start + index[1] > len(tokenized_to_original_index):
                yield {
                    'example': example,
                    'long_answer': [-1, -1],
                    'short_answer': [-1, -1],
                    'yes_no_answer': class_pred
                }
                continue
            short_start_index = tokenized_to_original_index[doc_start + index[0]]
            short_end_index = tokenized_to_original_index[doc_start + index[1]]
            long_start_index = -1
            long_end_index = -1
            for candidate in example.candidates:
                if candidate['start_token'] <= short_start_index and short_end_index <= candidate['end_token']:
                    long_start_index = candidate['start_token']
                    long_end_index = candidate['end_token']
                    break
            yield {
                'example': example,
                'long_answer': [long_start_index, long_end_index],
                'short_answer': [short_start_index, short_end_index],
                'yes_no_answer': class_pred
            }

    def end(self) -> Dict[str, Dict]:
        """Get predictions in submission format.
        """
        preds = {}
        for pred in self._generate_predictions():
            example = pred['example']
            long_start_index, long_end_index = pred['long_answer']
            short_start_index, short_end_index = pred['short_answer']
            class_pred = pred['yes_no_answer']

            long_answer = f'{long_start_index}:{long_end_index}' if long_start_index != -1 else np.nan
            short_answer = f'{short_start_index}:{short_end_index}'
            class_pred = self.class_labels[class_pred.argmax()]
            short_answer += ' ' + class_pred if class_pred in ['YES', 'NO'] else ''
            preds[f'{example.example_id}_long'] = long_answer
            preds[f'{example.example_id}_short'] = short_answer
        return preds

    def score(self) -> Dict[str, float]:
        """Calculate score of all examples.
        """

        def _safe_divide(x: int, y: int) -> float:
            """Compute x / y, but return 0 if y is zero.
            """
            if y == 0:
                return 0.
            else:
                return x / y

        def _compute_f1(answer_stats: List[List[bool]]) -> float:
            """Computes F1, precision, recall for a list of answer scores.
            """
            has_answer, has_pred, is_correct = list(zip(*answer_stats))
            precision = _safe_divide(sum(is_correct), sum(has_pred))
            recall = _safe_divide(sum(is_correct), sum(has_answer))
            print('precision=', precision)
            print('recall=', recall)
            f1 = _safe_divide(2 * precision * recall, precision + recall)
            return f1

        long_scores = []
        short_scores = []
        for pred in self._generate_predictions():
            example = pred['example']
            long_pred = pred['long_answer']
            short_pred = pred['short_answer']
            class_pred = pred['yes_no_answer']
            yes_no_label = self.class_labels[class_pred.argmax()]

            if (DEBUG):
                print(example.example_id)
                print('long_pred=', long_pred)
                print('short_pred=', short_pred)
                #logging.info('long_pred=%d',long_pred)
                #logging.info("%s",example.question_text)
                answer="answer:"
                for i in range(long_pred[0], long_pred[1]):
                    answer += example.doc_tokens[i]
                #logging.info("%s", answer)
                print(answer)
            # long score
            long_label = example.annotations['long_answer']
            has_answer = long_label['candidate_index'] != -1
            has_pred = long_pred[0] != -1 and long_pred[1] != -1
            is_correct = False
            if long_label['start_token'] == long_pred[0] and long_label['end_token'] == long_pred[1]:
                is_correct = True
            long_scores.append([has_answer, has_pred, is_correct])

            # short score
            short_labels = example.annotations['short_answers']
            class_pred = example.annotations['yes_no_answer']
            has_answer = yes_no_label != 'NONE' or len(short_labels) != 0
            has_pred = class_pred != 'NONE' or (short_pred[0] != -1 and short_pred[1] != -1)
            is_correct = False
            if class_pred in ['YES', 'NO']:
                is_correct = yes_no_label == class_pred
            else:
                for short_label in short_labels:
                    if short_label['start_token'] == short_pred[0] and short_label['end_token'] == short_pred[1]:
                        is_correct = True
                        break
            short_scores.append([has_answer, has_pred, is_correct])

        print('Long Answer score out of ', len(long_scores))
        long_score = _compute_f1(long_scores)
        print('Short Answer')
        short_score = _compute_f1(short_scores)
        return {
            'long_score': long_score,
            'short_score': short_score,
        }

# Rekha added
# EVAL STARTING
data_reader = JsonChunkReader(EVAL_DATA_PATH, convert_func, chunksize=chunksize)

# In[ ]:


eval_start_time = time.time()
valid_data = next(data_reader)
valid_data = list(itertools.chain.from_iterable(valid_data))
valid_dataset = Subset(valid_data, range(len(valid_data)))
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn)
valid_scores = eval_model(qa_model, classifier_model, valid_loader, device=device)

print(f'calculate validation score done in {(time.time() - eval_start_time) / 60:.1f} minutes.')

# In[ ]:


long_score = valid_scores['long_score']
short_score = valid_scores['short_score']
overall_score = valid_scores['overall_score']
print('validation scores:')
print(f'\tlong score    : {long_score:.4f}')
print(f'\tshort score   : {short_score:.4f}')
print(f'\toverall score : {overall_score:.4f}')
print(f'all process done in {(time.time() - start_time) / 3600:.1f} hours.')