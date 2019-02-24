# -*- coding:utf-8 -*-
import argparse

from config import save_dir, corpus, corpus_name, datafile
from train import trainIters
from evaluate import runTest

def parse():
    parser = argparse.ArgumentParser(description="Attention seq2seq model of ChatBot")
    parser.add_argument('-tr', '--train', action='store_true', default=False, help="train model")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="test model")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help="Learning rate for encode and decode")
    parser.add_argument('-mf', '--model_file', default='',help="specify the model file to load for training or test")
    parser.add_argument('-df', '--data_file', default=datafile, help="the data file fed to train")
    parser.add_argument('-la', '--layers', type=int, default=2, help="hidden layers for encode and decode")
    parser.add_argument('-hd', '--hidden_size', type=int, default=500, help="hidden size of encode and decode")
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout value')
    parser.add_argument('-at', '--attn_model', default='dot', help='attention model method, [dot, general, contact]')
    parser.add_argument('-it', '--iteration', type=int, default=40000, help="iteration num for training")
    parser.add_argument('-bt', '--batch_size', type=int, default=64, help="batch size of training data")
    parser.add_argument('-cl', '--clip', type=float, default=50.0, help='threshold for avoiding gradient explode')
    parser.add_argument('-dl', '--decoder_learning_ratio', type=float, default=5.0, help="learning ratio for decoder")
    parser.add_argument('-s', '--save_every', type=int, default=1000, help="save checkpoint every s iterations")
    parser.add_argument('-p', '--print_every', type=int, default=100, help="print train result every s iterations")
    parser.add_argument('-bs', '--beam_size', type=int, default=1, help="beam size for decoder search")
    parser.add_argument('-in', '--input', action='store_true', default=False, help="Test the model by input sentence")

    args = parser.parse_args()
    return args

def run(args):
    learning_rate, loadFilename, datafile, decoder_n_layers,\
    encoder_n_layers, hidden_size, dropout, attn_model, \
    n_iteration, batch_size, save_every, print_every, \
    decoder_learning_ratio, clip, beam_size, inp = args.learning_rate, args.model_file, args.data_file, args.layers, \
                                   args.layers, args.hidden_size, args.dropout, args.attn_model, args.iteration, \
                                   args.batch_size, args.save_every, args.print_every, \
                                   args.decoder_learning_ratio, args.clip, args.beam_size, args.input
    if args.test:
        if loadFilename:
            print("Starting testing model!")
            runTest(decoder_n_layers, hidden_size, False, loadFilename, beam_size, inp, datafile)
        else:
            raise RuntimeError("Please assign modelFile to load")
    elif args.train:
        print("Starting Training model!")
        trainIters(attn_model=attn_model, hidden_size=hidden_size, encoder_n_layers=encoder_n_layers, \
                   decoder_n_layers=decoder_n_layers, save_dir=save_dir, n_iteration=n_iteration, batch_size=batch_size, \
                   learning_rate=learning_rate, decoder_learning_ratio=decoder_learning_ratio, print_every=print_every, \
                   save_every=save_every, clip=clip, dropout=dropout, corpus_name=corpus_name, datafile=datafile, \
                   modelFile=loadFilename)
    else:
        raise RuntimeError("Please specify a running mode between train and test")

if __name__ == "__main__":
    args = parse()
    run(args)