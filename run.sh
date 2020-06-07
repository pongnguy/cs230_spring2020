#!/bin/bash
#parser.add_argument("--data_dir", type=str, default='/data/global_data/rekha_data', help="directory containing training and validation files")
#parser.add_argument("--dropout", type=float, default=0.1, help="-")
#parser.add_argument("--train_size", type=int, default=100, help="number of training examples")
#parser.add_argument("--valid_size", type=int, default=10, help="number of validation examples")
#parser.add_argument("--learning_rate", type=float, default=2e-5, help="initial learning rate")
#parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
#parser.add_argument("--batch_size", type=int, default=16, help="batch size to use for training examples")
#parser.add_argument("--fp16", type=bool, default=False, help="whether to use 16-bit precision")
#parser.add_argument("--hidden_layers", type=int, default=6, help="number of hidden layers from pretrained model to use")
#parser.add_argument("--frozen_layers", type=int, default=6, help="number of layers to freeze in pretrained model")
#parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer")
#parser.add_argument("--unknown_weight", type=float, default=0.3, help="weight of unknown label in loss function")
#parser.add_argument("--do_train", type=bool, default=True, help="do training or evaluate a trained model")

python3 -u train_classifier.py --train_size=100 --valid_size=10 --dropout=0.2
python3 -u train_classifier.py --train_size=100 --valid_size=10 --dropout=0.3
