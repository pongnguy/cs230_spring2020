from typing import Callable, Dict, List, Generator, Tuple

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader
import time
import itertools
import functools

from tqdm import tqdm

from train_classifier import DistilBertForQuestionAnswering, JsonChunkReader, Result, TextDataset
from train_classifier import convert_data #convert_func
from train_classifier import num_labels, output_model_file, EVAL_DATA_PATH, eval_collate_fn, collate_fn, batch_size, chunksize, max_seq_len, max_question_len, doc_stride #, global_step


#num_labels = 2 #5


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
        for inputs, examples, example_ids, text_labels in tqdm(valid_loader):
            input_ids, attention_mask, token_type_ids = inputs
            y_batch = (y.to(device) for y in examples)

            #y_preds = model(input_ids.to(device),
            #                attention_mask.to(device),
            #                token_type_ids.to(device))
            y_preds = model(input_ids.to(device),
                           attention_mask=attention_mask.to(device))

            #start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds) # alfred our model only does classification
            _, _, class_preds = y_preds
            _, _, class_labels = y_batch
            #start_logits, start_index = torch.max(start_preds, dim=1)
            #end_logits, end_index = torch.max(end_preds, dim=1)

            # TODO alfred need to fix this code to work with just classifier

            # span logits minus the cls logits seems to be close to the best
            cls_logits = start_preds[:, 0] + end_preds[:, 0]  # '[CLS]' logits
            logits = start_logits + end_logits - cls_logits  # (batch_size,)
            indices = torch.stack((start_index, end_index)).transpose(0, 1)  # (batch_size, 2)
            result.update(examples, logits.numpy(), indices.numpy(), class_preds.numpy())

    return result.score()



# Put everything inside this if statement to prevent multiprocessing error on windows
if __name__ == '__main__':
    # In[ ]:

    # use GPU if it is available
    if torch.cuda.is_available():
        print('use cuda device')
        device = torch.device('cuda')
    else:
        print('use cpu device')
        device = torch.device('cpu')

    # In[ ]:


    # load an existing model
    # ------
    # create model
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased-distilled-squad')
    #print('config file', config)
    config.num_labels = num_labels
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad',
                                                           config=config)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)

    convert_func = functools.partial(convert_data,   # returns an Example class
                                         tokenizer=tokenizer,
                                         max_seq_len=max_seq_len,
                                         max_question_len=max_question_len,
                                         doc_stride=doc_stride)

    # load saved weights
    model.load_state_dict(torch.load(output_model_file))

    # Rekha added
    # EVAL STARTING
    data_reader = JsonChunkReader(EVAL_DATA_PATH, convert_func, chunksize=chunksize)

    # In[ ]:


    eval_start_time = time.time()
    valid_data = next(data_reader)
    #valid_data = list(itertools.chain.from_iterable(valid_data))
    #valid_dataset = Subset(valid_data, range(len(valid_data)))
    valid_dataset = TextDataset(valid_data)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #eval_collate_fn)
    valid_scores = eval_model(model, valid_loader, device=device)

    print(f'calculate validation score done in {(time.time() - eval_start_time) / 60:.1f} minutes.')

    # In[ ]:


    long_score = valid_scores['long_score']
    short_score = valid_scores['short_score']
    overall_score = valid_scores['overall_score']
    print('validation scores:')
    print(f'\tlong score    : {long_score:.4f}')
    print(f'\tshort score   : {short_score:.4f}')
    print(f'\toverall score : {overall_score:.4f}')
    #print(f'all process done in {(time.time() - start_time) / 3600:.1f} hours.')

    # In[ ]:


    #get_ipython().system('wc -l $DATA_PATH')
    #get_ipython().system('wc -l $EVAL_DATA_PATH')

    # In[ ]:

    #print(f'trained {global_step * batch_size} samples')
    #print(f'training time: {(time.time() - start_time) / 3600:.1f} hours')

    # In[ ]:



