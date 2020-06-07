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

#python3 -u train_classifier.py --learning_rate=1e-10 --train_size=10000 --valid_size=100 | tee out1.txt 2>&1
#python3 -u train_classifier.py --unknown_weight=0.4 --train_size=10000 --valid_size=100 | tee out2.txt 2>&1
#python3 -u train_classifier.py --learning_rate=1e-8 --train_size=10000 --valid_size=100 | tee out3.txt 2>&1
#python3 -u train_classifier.py --unknown_weight=0.5 --train_size=10000 --valid_size=100 | tee out4.txt 2>&1
#python3 -u train_classifier.py --learning_rate=1e-6 --train_size=10000 --valid_size=100 | tee out5.txt 2>&1
#python3 -u train_classifier.py --unknown_weight=0.6 --train_size=10000 --valid_size=100 | tee out6.txt 2>&1
#python3 -u train_classifier.py --learning_rate=1e-4 --train_size=10000 --valid_size=100 | tee out7.txt 2>&1
#python3 -u train_classifier.py --learning_rate=1e-3 --train_size=10000 --valid_size=100 | tee out8.txt 2>&1

python3 -u train_classifier.py --stage=both --unknown_weight=0.3 --learning_rate=1e-5 --train_size=10000 --valid_size=100 | tee out9_lr1e5.txt 2>&1
python3 -u train_classifier.py --stage=both --unknown_weight=0.3 --learning_rate=4e-5 --train_size=10000 --valid_size=100 | tee out10_lr2e5.txt 2>&1
python3 -u train_classifier.py --stage=both --unknown_weight=0.3 --learning_rate=1e-6 --epochs=4 --train_size=10000 --valid_size=100 | tee out10_lr2e5.txt 2>&1
python3 -u train_classifier.py --stage=both --unknown_weight=0.3 --learning_rate=1e-5 --epochs=4 --train_size=10000 --valid_size=100 | tee out9_lr1e5.txt 2>&1
python3 -u train_classifier.py --stage=both --unknown_weight=0.3 --learning_rate=1e-5 --train_size=10000 --valid_size=2000 | tee out9_lr1e5.txt 2>&1
python3 -u train_classifier.py --stage=both --unknown_weight=0.3 --learning_rate=2e-5 --train_size=10000 --valid_size=2000 | tee out10_lr2e5.txt 2>&1