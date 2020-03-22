'''
date: 2020-03-17

创建词汇表，并将qa对话加载到内存
1. 获得{word:index},{index:word},{word:count}
2. 处理pairs: 转小写，去标点符号，过滤低频词对应的pairs和超出最大长度的pairs
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script,trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

#------------------------对词进行处理，包括{word：index},{index:word},{word:count}-------------------
# 默认词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}  # word:index
        self.word2count = {}  # word:count
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}  # index:word
        self.num_words = 3  # Count SOS, EOS, PAD

    # 添加新词
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于特定计数阈值的单词
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 重初始化字典
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

#------------------对pairs进行处理-------------------------------------------------------

MAX_LENGTH = 10  # 输入句子的最大长度，长于该长度的截断或删掉，短于该长度的padding

# 将Unicode字符串转换为纯ASCII，多亏了
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 大写字母变小写，去掉标点符号
def normalizeString(one_str):
    new_str = one_str.lower()
    new_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", " ",new_str)  

    return new_str



# 初始化Voc对象 和 格式化pairs对话存放到list中
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# 如果对 'p' 中的两个句子都低于 MAX_LENGTH 阈值，则返回True
# 过滤掉超过最大长度的问答句子对，只要有一个超过都删除掉
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 过滤满足条件的 pairs 对话
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



# 使用上面定义的函数，返回一个填充的voc对象和对列表
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


#------------------------过滤频率比较低的词------------------------------------

MIN_COUNT = 3 # 词频低于3的词去掉

def trimRareWords(voc,pairs,MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep_input = True
        keep_output = True

        # input
        for word in input_sentence.split(" "):
            if word not in voc.word2index:
                keep_input = False
                break

        # output
        for word in output_sentence.split(" "):
            if word not in voc.word2index:
                keep_output = False
                break
        # 比较严格，词频低的对应的pair也不要了
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


#

if __name__ == "__main__":

    save_dir = os.path.join("data","save")
    corpus = "cornell movie-dialogs corpus"
    corpus_name = "cornell movie-dialogs corpus"
    data_file = os.path.join("data",corpus_name,"formatted_movie_lines.txt")
    voc, pairs = loadPrepareData(corpus, corpus_name, data_file,save_dir)

    print("\n pairs: ")
    for pair in pairs[:10]:
        print(pair)

    # 修剪voc和对
    pairs = trimRareWords(voc, pairs, MIN_COUNT)