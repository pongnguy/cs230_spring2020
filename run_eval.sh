#!/bin/bash

#parser.add_argument("--data_dir", type=str, default='/data/global_data/rekha_data', help="directory containing training and validation files")
#parser.add_argument("--classifier_model_dir", type=str, default='/data/sv/distilbert', help="model file")
#parser.add_argument("--classifier_dropout", type=float, default=0.1, help="-")
#parser.add_argument("--qa_dropout", type=float, default=0.1, help="-")
#parser.add_argument("--valid_size", type=int, default=100, help="number of validation examples")
#parser.add_argument("--batch_size", type=int, default=16, help="batch size to use for training examples")
#parser.add_argument("--fp16", type=bool, default=False, help="whether to use 16-bit precision")
#parser.add_argument("--hidden_layers", type=int, default=6, help="number of hidden layers from pretrained model to use")

echo $PWD
python3 -u eval.py --classifier_model_dir="/data/sv/valid100_dr_0.1_unkWt_0.5lr2e-0507-06_07_50_47"  --classifier_dropout=0.1 |tee -a eval1.txt 2>&1
