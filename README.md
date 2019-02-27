## chatbot

**This is a pytorch seq2seq exersice,  major code comes  from [pytorch-tutorial-chatbot_tutorial.](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)**

**Beam search method was implemented to handle encoder-decoder output result, that works well on long sentence input.** 

## Requirement

- python 3.6+
- pytorch 1.0.0
- tqdm

## corpus file 

In the corpus file, the input-output sequence pairs should be in the same line split with '\t' . For example,

```
She okay?	I hope so.
They do to!	They do not!
```

The corpus files should be placed under a path like,

```
chatbot/data/corpus_name/corpus_file_name
```

Otherwise, the corpus file will be tracked by git.

## Run

#### Print Parameters

`./run.py --help`

~~~
optional arguments:
  -h, --help            show this help message and exit
  -tr, --train          train model
  -te, --test           test model
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for encode and decode
  -mf MODEL_FILE, --model_file MODEL_FILE
                        specify the model file to load for training or test
  -df DATA_FILE, --data_file DATA_FILE
                        the data file fed to train
  -la LAYERS, --layers LAYERS
                        hidden layers for encode and decode
  -hd HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        hidden size of encode and decode
  -d DROPOUT, --dropout DROPOUT
                        dropout value
  -at ATTN_MODEL, --attn_model ATTN_MODEL
                        attention model method, [dot, general, contact]
  -it ITERATION, --iteration ITERATION
                        iteration num for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size of training data
  -cl CLIP, --clip CLIP
                        threshold for avoiding gradient explode
  -dl DECODER_LEARNING_RATIO, --decoder_learning_ratio DECODER_LEARNING_RATIO
                        learning ratio for decoder
  -s SAVE_EVERY, --save_every SAVE_EVERY
                        save checkpoint every s iterations
  -p PRINT_EVERY, --print_every PRINT_EVERY
                        print train result every s iterations
  -bs BEAM_SIZE, --beam_size BEAM_SIZE
                        beam size for decoder search
  -in, --input          Test the model by input sentence
~~~

#### Training

Run this command to start training, change the argument values in your own need.

```
python run.py -tr -df <CORPUS_FILE_PATH>  -la 1 -hi 512 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```

Continue training with saved model.

```
python run.py -tr -df <CORPUS_FILE_PATH> -mf <MODEL_FILE_PATH> -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```

For more options,

```
python run.py -h
```

#### Testing

Models will be saved in `chatbot/save/model` while training, and this can be changed in `config.py`.
Evaluate the saved model with input sequences in the corpus.

```
python run.py -te -mf <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH>
```

Test the model with input sequence manually.

```
python main.py -te <MODEL_FILE_PATH> -in
```

Beam search with size k.

```
python run.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -bs k [-in]
```
