"""
General file for inputting the data

"""
from gzip import GzipFile

from tqdm import tqdm
import json
import numpy as np
#from typing import TypedDict, List
import tensorflow as tf


def format_KaggleToSquad(dataset_kaggle):

    #Initialize the squad data dictionary
    dataset_Squad = {'version': 'v2.0', 'data': []}
    exampleCount = 0

    for example in dataset_kaggle['lines']:
        dataset_Squad['data'].append({'title': example['document_url'], 'paragraphs': []})
        for annotation in example['annotations']:  # annotations is an array in Kaggle, however it only ever has one element in the training data
            dataset_Squad['data'][-1]['paragraphs'].append({'qas': [{'question': example['question_text'], 'id': annotation['annotation_id'], 'is_impossibe': False if (len(annotation['short_answers'])>0) else True, 'answers': []}], 'context': example['document_text'], 'long_answer_candidates': example['long_answer_candidates']})
            for short_answer in annotation['short_answers']:   # only a single question gets copied since Kaggle dataset only has a single question per training example, however I am still indexing the last element in the qas array
                curAnswer = dataset_Squad['data'][-1]['paragraphs'][-1]['qas'][-1]['answers']
                curAnswer.append({'text': 'TODO', 'answer_start': short_answer['start_token'], 'answer_end': short_answer['end_token']})

    return dataset_Squad

def random_sample_negative_candidates(distribution):
    temp = np.random.random()
    value = 0.
    for index in range(len(distribution)):
        value += distribution[index]
        if value > temp:
            break
    return index

def jsonlToJson(json_dir, max_data = 9999999999):
    """
    Inputting the data from Kaggle

    :param json_dir:  path to the filename
    :type json_dir:  str
    :param max_data:  maximum number of rows to input
    :type max_data:  int
    :returns: processed dataset
    :rtype: DatasetKaggle
    """

    # prepare input
    #json_dir = '../../input/simplified-nq-train.jsonl'
    #max_data = 9999999999

    id_list = []
    data_dict =  { "lines": [] }  #DatasetKaggle # alfred
    n = 0
    with open(json_dir, 'r', 1) as f:
    #with open(json_dir, 'r', 1, 'utf16') as f:
        n += 1
        if n % 100 == 0:
            print(n)

        #iterable = enumerate(f)
        #tqdm_val = tqdm(iterable)
        #for n, line in tqdm_val:
            #print(n)
        for n, line in enumerate(f):
            #line = f.readline()
            if n >= max_data:
                break
            line_json = json.loads(line)
            data_dict['lines'].append(line_json)

    return data_dict


"""
    if isinstance(json_dir, str):
        gzipped_input_file = open(json_dir, 'rb')
    #logging.info('parsing %s ..... ', gzipped_input_file.name)
    #annotation_dict = {}
    with GzipFile(fileobj=gzipped_input_file) as input_file:
        for line in input_file:
            json_example = json.loads(line)
            data_dict.append(json_example)
"""

def inputdata_KaggleWinner(json_dir, max_data = 9999999999):
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

            if is_pos and len(data['long_answer_candidates']) > 1:
                data_id = data['example_id']
                id_list.append(data_id)

                # uniform sampling
                distribution = np.ones((len(data['long_answer_candidates']),), dtype=np.float32)
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

    return data_dict




def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

            # Run callback
            output_fn(feature)

            unique_id += 1


def compute_lengthHistogram(datasetKaggle, numBins):
    """
        Outputs various statistics of the Kaggle dataset

        :param datasetKaggle: the input dataset formatted as a dictionary
        :type datasetKaggle: dict
        :return: dictionary with the processed information
        :rtype: dict


    """


    maxLength = 0
    minLength = 9999999999

    for example in datasetKaggle: # first pass through data to determine the mix and max length
        maxLength = max(len(example['document_text']), maxLength)
        minLength = min(len(example['document_text']), minLength)

    """for example in datasetKaggle:
        # TODO implement code to do the binning"""

    return {'max_length': maxLength, 'min_length': minLength}






def compute_statistics(datasetKaggle):
    """
    Outputs various statistics of the Kaggle dataset

    :param datasetKaggle: the input dataset formatted as a dictionary
    :type datasetKaggle: dict
    :return: dictionary with the processed information
    :rtype: dict


    """

    yesNoAnswer = 0
    annotationsMax = 0
    averageLength = 0
    totalExamples = len(datasetKaggle)

    for example in datasetKaggle:
        annotationsMax = max(len(example['annotations']), annotationsMax)  # check for the maximum number of annotations
        if example['annotations'][0]['yes_no_answer'] != 'NONE':
            yesNoAnswer += 1

        averageLength += len(example['document_text']) / totalExamples
    output = {'annotationsMax': annotationsMax, 'num_yesNo': yesNoAnswer, 'text_avgLength': averageLength}
    return output



