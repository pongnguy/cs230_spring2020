import argparse
from datetime import datetime

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
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
import sys
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
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

import pdb
import json
import pickle
from os import path

#import asyncio
#import aiofiles as aiof


def print_both(file, *args):
    temp = sys.stdout #assign console output to a variable
    print(' '.join([str(arg) for arg in args]))
    sys.stdout = file
    print(' '.join([str(arg) for arg in args]))
    file.flush()  # force flush to see output in file on disk
    sys.stdout = temp #set stdout back to console output







# In[ ]:


@dataclass
class Example(object):
    example_id: int
    candidates: List[Dict]
    annotations: Dict
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
            with Pool(10) as p:
                obj = p.map(self.convert_data, lines)
                # print('Length of obj', len(obj))
            return obj

        self.close()
        raise StopIteration


# In[ ]:


class TextDataset(Dataset):
    """Dataset for [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering).

    Parameters
    ----------
    examples : list of Example
        The whole Dataset.
    """

    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        annotated = list(
            filter(lambda example: example.class_label != 'unknown', self.examples[index]))
        if len(annotated) == 0:
            return random.choice(self.examples[index])
        return random.choice(annotated)


def collate_fn(examples: List[Example]) -> List[List[torch.Tensor]]:
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

    # output labels
    all_labels = ['long', 'no', 'short', 'unknown', 'yes']
    start_positions = np.array([example.start_position for example in examples])
    end_positions = np.array([example.end_position for example in examples])
    class_labels = [all_labels.index(example.class_label) for example in examples] # alfred class label as the index
    start_positions = np.where(start_positions >= max_len, -1, start_positions)
    end_positions = np.where(end_positions >= max_len, -1, end_positions)
    labels = [torch.LongTensor(start_positions),
              torch.LongTensor(end_positions),
              torch.LongTensor(class_labels)]

    return [inputs, labels]

class DistilBertForQuestionAnswering(DistilBertModel):
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
        super(DistilBertForQuestionAnswering, self).__init__(config)
        self.bert = DistilBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(args.dropout) #RekhaDist
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            #token_type_ids=token_type_ids,
                            #position_ids=position_ids,
                            #head_mask=head_mask
                            )

        #print('Outputs shape=', outputs.shape)
        sequence_output = outputs[0] # these are last hidden states
        #print("sequence_output type",sequence_output.type())
        pooled_output = sequence_output[:,0,:] #Rekha hack check # alfred returns the hidden state of the [CLS] token which is taken as fixed-dimensional pooled representation of the input sequence



        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        start_dummy = torch.zeros(batch_size, max_seq_len)
        end_dummy = torch.zeros(batch_size, max_seq_len)
        return start_dummy, end_dummy, classifier_logits

#['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']

def loss_fn_classifier(preds, labels):
    _, _, class_preds = preds
    _, _, class_labels = labels

    class_weights = [1.0, 1.0, 1.0, args.unknown_weight, 1.0]
    class_weights = torch.FloatTensor(class_weights).cuda()
    class_loss = nn.CrossEntropyLoss(class_weights)(class_preds, class_labels)

    return class_loss

def eval_model(
        model: nn.Module,
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
    model.to(device)
    #model.half()
    model.eval()
    with torch.no_grad():
        result = Result()
        for inputs, examples in tqdm(valid_loader):
            input_ids, attention_mask, token_type_ids = inputs
            y_preds = model(input_ids.to(device),
                            attention_mask.to(device),
                            token_type_ids.to(device))

            _, _, class_preds = (p.detach().cpu() for p in y_preds)
            # start_logits, start_index = torch.max(start_preds, dim=1)
            # end_logits, end_index = torch.max(end_preds, dim=1)

            # span logits minus the cls logits seems to be close to the best
            # cls_logits = start_preds[:, 0] + end_preds[:, 0]  # '[CLS]' logits
            #logits = start_logits + end_logits - cls_logits  # (batch_size,)
            #indices = torch.stack((start_index, end_index)).transpose(0, 1)  # (batch_size, 2)
            #result.update(examples, np.array(list(class_preds)))
            #result.update(examples, class_preds.numpy())

            result.update(examples, class_preds.numpy()) # add the predictions to the examples

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
            # logits: torch.Tensor,
            # indices: torch.Tensor,
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
            # if self.is_valid_index(example, indices[i]) and self.best_scores[example.example_id] < logits[i]:
            if True:
                #self.best_scores[example.example_id] = logits[i]
                key_id = str(example.example_id)+'_'+str(example.doc_start)
                self.examples[key_id] = example # BUG this will overwrite previous chunks having the same example_id
                self.results[key_id] = [
                    example.doc_start, class_preds[i]]  # Store class_preds in results # alfred these are raw numbers

    def _generate_predictions(self) -> Generator[Dict, None, None]:
        """Generate predictions of each examples.
        """
        for example_id_doc_start in self.results.keys():
            _, class_pred = self.results[example_id_doc_start]
            example = self.examples[example_id_doc_start]
            #tokenized_to_original_index = example.tokenized_to_original_index

            #short_start_index = tokenized_to_original_index[doc_start + index[0]]
            #short_end_index = tokenized_to_original_index[doc_start + index[1]]
            #long_start_index = -1
            #long_end_index = -1
            #for candidate in example.candidates:
            #    if candidate['start_token'] <= short_start_index and short_end_index <= candidate['end_token']:
            #        long_start_index = candidate['start_token']
            #        long_end_index = candidate['end_token']
            #       break
            yield {
                'example': example,
                #'long_answer': [-1, -1],
                #'short_answer': [-1, -1],
                'class_label': class_pred
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
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            num_has_answer = 0
            for ha, hp in zip(has_answer, has_pred):
                if(hp and ha):
                    tp += 1
                    num_has_answer += 1
                elif(hp and not ha):
                    fp += 1
                elif(not hp and ha):
                    fn += 1
                    num_has_answer += 1
                elif(not hp and not ha):
                    tn += 1

                #if(ha and hp and is_correct):
                #    tp += 1
                #elif(ha and hp and not is_correct):
                #    fp += 1
                #elif(not ha and not hp and is_correct):
                #    tn += 1
                #elif(not ha and not hp and not is_correct):
                #    fn += 1
                #elif(hp and not ha):
                #    fp = fp + 1
                #elif(ha and not hp):
                #    fn = fn + 1

            precision = _safe_divide(tp, tp + fp)
            recall = _safe_divide(tp, tp + fn)
            print_both(out_file, 'tp=',tp, '  fp=', fp)    #    Confusion matrix
            print_both(out_file, 'fn=', fn, '  tn=', tn)  # Confusion matrix
            print_both(out_file,'precision=', precision)
            print_both(out_file,'recall=', recall)
            print_both(out_file, 'num has answer=', num_has_answer)

            f1 = _safe_divide(2 * precision * recall, precision + recall)
            return f1

        classification_scores = []
        long_scores = []
        short_scores = []
        for pred in self._generate_predictions():
            example = pred['example']
            #long_pred = pred['long_answer']     # alfred this always returns [-1, -1]
            #short_pred = pred['short_answer']
            #class_pred = pred['yes_no_answer']
            #yes_no_label = self.class_labels[class_pred.argmax()] # alfred what is this here for?
            class_pred = pred['class_label']
            class_pred = self.class_labels[class_pred.argmax()]

            if (DEBUG):
                print_both(debug_file,example)
                print_both(debug_file, example.doc_start)
                print_both(debug_file, example.class_label)
                print_both(debug_file,'prediction '+class_pred)
                #print_both(out_file,yes_no_label)
                #logging.info('long_pred=%d',long_pred)
                #logging.info("%s",example.question_text)
                #logging.info("%s", answer)
            # long score





            # Actual ans
            #long_label = example.annotations['long_answer']
            #has_answer = long_label['candidate_index'] != -1        # input labels true or false for long ans part

            # predicted output
            #has_pred = yes_no_label != 'UNKNOWN'



            #is_correct = False
            #if long_label['start_token'] == long_pred[0] and long_label['end_token'] == long_pred[1]:
            #    is_correct = True  # BUG there is an error here
            has_pred = (class_pred.upper() != 'UNKNOWN')
            has_answer = (example.class_label.upper() != 'UNKNOWN')
            is_correct = (has_pred == has_answer)
            classification_scores.append([has_answer, has_pred, is_correct])  # alfred won't this always return false??

            # short score
            # short_labels = example.annotations['short_answers']
            # class_pred = example.annotations['yes_no_answer']
            # has_answer = yes_no_label != 'NONE' or len(short_labels) != 0
            # has_pred = class_pred != 'NONE' or (short_pred[0] != -1 and short_pred[1] != -1)
            # is_correct = False
            # if class_pred in ['YES', 'NO']:
            #     is_correct = yes_no_label == class_pred
            # else:
            #     for short_label in short_labels:
            #         if short_label['start_token'] == short_pred[0] and short_label['end_token'] == short_pred[1]:
            #             is_correct = True
            #             break
            # short_scores.append([has_answer, has_pred, is_correct])

        print_both(out_file,'Classification Answer')
        classification_score = _compute_f1(classification_scores)
        # print('Short Answer')
        # short_score = _compute_f1(short_scores)
        return {
            'classification_score': classification_score,
            # 'short_score': short_score,
            #'overall_score': (long_score + short_score) / 2
        }

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

# Put everything inside this if statement to prevent multiprocessing error on windows
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='/data/global_data/rekha_data',
                        help="directory containing training and validation files")
    parser.add_argument("--dropout", type=float, default=0.1, help="-")
    parser.add_argument("--train_size", type=int, default=10000, help="number of training examples")
    parser.add_argument("--valid_size", type=int, default=10, help="number of validation examples")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size to use for training examples")
    parser.add_argument("--fp16", type=bool, default=False, help="whether to use 16-bit precision")
    parser.add_argument("--hidden_layers", type=int, default=6,
                        help="number of hidden layers from pretrained model to use")
    parser.add_argument("--frozen_layers", type=int, default=6, help="number of layers to freeze in pretrained model")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer")
    parser.add_argument("--unknown_weight", type=float, default=0.3, help="weight of unknown label in loss function")
    parser.add_argument("--do_train", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, help="do training or evaluate a trained model")
    parser.add_argument("--eval_model_file", type=str, default="", help="path to saved model")

    args = parser.parse_args()
    train_size = args.train_size
    valid_size = args.valid_size

    DATA_PATH = os.path.join(args.data_dir, 'simplified-nq-train.' + str(train_size) + '.jsonl')
    EVAL_DATA_PATH = os.path.join(args.data_dir, 'simplified-nq-valid.' + str(valid_size) + '.jsonl')

    chunksize = 1000

    #
    # get_ipython().system('wc -l $DATA_PATH')
    # get_ipython().system('wc -l $EVAL_DATA_PATH')
    DEBUG = True
    EVALUATION = not args.do_train

    # In[ ]:

    # DATA_DIR = Path('../input/tensorflow2-question-answering/')
    # DATA_PATH = DATA_DIR / 'simplified-nq-train.jsonl'

    start_time = time.time()

    seed = 1029

    max_seq_len = 384
    max_question_len = 64
    doc_stride = 128

    num_labels = 5
    n_epochs = args.epochs
    lr = args.learning_rate
    warmup = 0.05
    batch_size = args.batch_size
    accumulation_steps = 4

    device = torch.device('cuda')
    output_dir = "dr" + str(args.dropout) + "unkWt" + str(args.unknown_weight) + "lr" + str(args.learning_rate) + datetime.now().strftime("%d-%m_%H_%M_%S")
    os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
    output_optimizer_file = os.path.join(output_dir, 'pytorch_optimizer.bin')
    output_amp_file = os.path.join(output_dir, 'pytorch_amp.bin')

    out_file = open(os.path.join(output_dir, 'results.txt'), 'w')
    if DEBUG:
        debug_file = open(os.path.join(output_dir, 'debug.txt'), 'w')

    with open(os.path.join(output_dir, 'hyperparameters.txt'), 'w') as hyperparameters_file:
        print(args, file=hyperparameters_file)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    #args = parser.parse_args()

    # RekhaDist
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased-distilled-squad')
    config.num_labels = 5
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad', config=config)

    model = model.to(device)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
    convert_func = functools.partial(convert_data,
                                     tokenizer=tokenizer,
                                     max_seq_len=max_seq_len,
                                     max_question_len=max_question_len,
                                     doc_stride=doc_stride)



    if not EVALUATION:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        num_train_optimization_steps = int(n_epochs * train_size / batch_size / accumulation_steps)
        print_both(out_file,'num_train_optimization_steps=', num_train_optimization_steps)
        num_warmup_steps = int(num_train_optimization_steps * warmup)
        print_both(out_file,'num_warmup_steps', num_warmup_steps)

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps)

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        model.zero_grad()
        model.train()


        data_reader = JsonChunkReader(DATA_PATH, convert_func, chunksize=chunksize)

        # saves all the examples if they do not already exist
        seq = 0
        assert(chunksize == 1000)  # hardcoding this
        # TODO check there exist pickles up to the requested training size (i.e. seq = training / chunksize)
        if not path.exists(f'final_project/shreyas/pickles/examples_chunk={chunksize}_seq={"%03d" % seq}.pickle'):
            print('start reading training set from json file', time.time())
            for examples in data_reader:
                print('end reading', time.time())
                with open(f'final_project/shreyas/pickles/examples_chunk={chunksize}_seq={"%03d" % seq}.pickle',
                          'wb') as f:  # save variable to binary file
                    pickle.dump(examples, f)
                examples_idx_max = seq
                seq += 1
                print('start reading training set from json file', time.time())
        else:
            # TODO get highest seq number starting from 0
            print('pickle files exist', time.time())
            while path.exists(f'final_project/shreyas/pickles/examples_chunk={chunksize}_seq={"%03d" % (seq + 1)}.pickle'):
                seq += 1

            examples_idx_max = seq
            assert (train_size % 1000 == 0 and train_size < 300000)  # checks if train_size is a multiple of 1000 since the pickles are chunksize 1000
            pickle_idx_max = int(train_size / 1000)

        global_step = 0
        print_both(out_file,'Rekha train_size=', train_size)
        print_both(out_file,'Rekha total=int(np.ceil(train_size / chunksize))=', int(np.ceil(train_size / chunksize)))
        print_both(out_file,'Rekha DATA_PATH', DATA_PATH)
        # print_both(out_file,'len(data_reader)',len(data_reader))
        for i in tqdm(range(pickle_idx_max)):
            # load in the pickle
            with open(f'final_project/shreyas/pickles/examples_chunk={chunksize}_seq={"%03d" % i}.pickle',
                      'rb') as f:  # save variable to binary file
                print('start load pickle file', time.time())
                examples = pickle.load(f)
                # alfred change all non 'unknown' labels to 'hasAnswer' in y_batch instead of regenerating the pickle files
                # for example_list in examples:
                #    for example in example_list:
                #        if example.class_label != 'unknown':
                #            example.class_label = 'hasAnswer'
                print('end load pickle file', time.time())

            train_dataset = TextDataset(examples)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            for x_batch, y_batch in train_loader:
                x_batch, attention_mask, token_type_ids = x_batch
                y_batch = (y.to(device) for y in y_batch)

                # RekhaDist
                # context = "The US has passed the peak on new coronavirus cases, " \
                #           "President Donald Trump said and predicted that some states would reopen this month." \
                #           "The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, " \
                #           "the highest for any country in the world."
                # questions = ["What was President Donald Trump's prediction?",
                #              "How many deaths have been reported from the virus?",
                #              "How many cases have been reported in the United States?"]
                # question_context_for_batch = []
                #
                # for question in questions:
                #     question_context_for_batch.append((question, context))
                # encoding = tokenizer.batch_encode_plus(question_context_for_batch, pad_to_max_length=True, return_tensors="pt")
                # input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
                # start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
                y_pred = model(x_batch.to(device),
                               attention_mask=attention_mask.to(device))

                #Rekha old
                # y_pred = model(x_batch.to(device),
                #                attention_mask=attention_mask.to(device),
                #                token_type_ids=token_type_ids.to(device))
                loss = loss_fn_classifier(y_pred, y_batch)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (global_step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                global_step += 1
            print_both(out_file,"Training loss:", loss)

            if (time.time() - start_time) / 3600 > 7:
                break

        del examples, train_dataset, train_loader
        gc.collect()

        torch.save(model.state_dict(), output_model_file)
        torch.save(optimizer.state_dict(), output_optimizer_file)
        torch.save(amp.state_dict(), output_amp_file)

        print_both(out_file,f'trained {global_step * batch_size} samples')
        print_both(out_file,f'training time: {(time.time() - start_time) / 3600:.1f} hours')




    # EVAL STARTING (Skip over training if EVAL = True)
    print_both(out_file,'EVAL STARTING')

    # load an existing model
    # ------
    # load saved weights

    eval_model_file = args.eval_model_file
    model.load_state_dict(torch.load(eval_model_file))


    data_reader = JsonChunkReader(EVAL_DATA_PATH, convert_func, chunksize=chunksize)

    eval_start_time = time.time()
    valid_data = next(data_reader)
    valid_data = list(itertools.chain.from_iterable(valid_data))
    valid_dataset = Subset(valid_data, range(len(valid_data)))
    print('len of valid_data from data_reader ', len(valid_data))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn)
    valid_scores = eval_model(model, valid_loader, device=device)

    print_both(out_file,f'calculate validation score done in {(time.time() - eval_start_time) / 60:.1f} minutes.')
    print_both(out_file,f'validation size {valid_size}')

    classification_score = valid_scores['classification_score']
    #short_score = valid_scores['short_score']
    print_both(out_file,'validation scores:')
    print_both(out_file,f'\tclassification score    : {classification_score:.4f}')
    #print_both(out_file,f'\tshort score   : {short_score:.4f}')
    print_both(out_file,f'all process done in {(time.time() - start_time) / 3600:.1f} hours.')

    #get_ipython().system('wc -l $DATA_PATH')
    #get_ipython().system('wc -l $EVAL_DATA_PATH')

    #print_both(out_file,f'trained {global_step * batch_size} samples')
    print_both(out_file,f'End of file: {(time.time() - start_time) / 3600:.1f} hours')


