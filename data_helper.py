# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unicodedata
import re
import itertools
import os
from config import *


# define fields name
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

acronym = ["can\'t", "cannot", "\'s", "\'ve ", "n\'t", "n\'t", "\'m", "\'re", "\'d", "\'ll",
           " e mail ", " e\-mail "]
acronym_rep = ["can not", "can not", " is", " have ", " not ", " not", " am", " are", " would",
               " will", " email ", " email "]

def printLines(file, n=10):
    # print("file path: {}".format(file))
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
        for line in lines[:n]:
            print(line)

#split each line into dictionary of fields
def Loadlines(fileName, fields):
    lines = {}
    with open(fileName, encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj["lineID"]] =lineObj

    return lines

def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding="iso-8859-1") as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj["utteranceIDs"])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            outputLine = conversation["lines"][i+1]["text"].strip()
            if inputLine and outputLine:
                qa_pairs.append([inputLine, outputLine])
    return qa_pairs

# data trim
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 #count PAD, SOS, EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("keep_words {} / total_words {} = {:.4f}".format(len(keep_words), len(self.word2index),\
                                                               len(keep_words)/len(self.word2index)))
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(clean_text(s)) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(datafile, corpus_name, needTrim = True):
    voc, pairs = readVocs(datafile, corpus_name)
    pairs = filterPairs(pairs)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    if needTrim:
        pairs = trimRareWords(voc, pairs, MIN_COUNT)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs


def loadPrepareData(datafile):
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData(datafile, corpus_name)

    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)

    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_length = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_length

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# only modify acronym and remove extra space
def clean_text(text):
    # acronym
    for idx in range(len(acronym)):
        if acronym[idx] in text:
            text = re.sub(acronym[idx], acronym_rep[idx], text)

    # remove extra space
    text = ' '.join(text.split())

    return text

# handle raw dataset and write into new file
def handle_raw_data(linesFile, convFile, lineFields, convFields, datafile):
    lines = {}
    conversations = []

    # load lines
    lines = Loadlines(linesFile, lineFields)

    # load conversations
    conversations = loadConversations(convFile, lines, convFields)

    # write sentence pairs into new file
    with open(datafile, 'w', encoding='utf-8') as f:
        for pair in extractSentencePairs(conversations):
            f.writelines([pair[0]+'\t', pair[1]+'\n'])


"""
linesFile = os.path.join(corpus, "movie_lines.txt")
convFile = os.path.join(corpus, "movie_conversations.txt")
datafile = os.path.join(corpus, "conversation.txt")
handle_raw_data(linesFile, convFile, MOVIE_LINES_FIELDS, MOVIE_CONVERSATIONS_FIELDS, datafile)
if __name__ == "__main__":
    loadPrepareData("")
"""