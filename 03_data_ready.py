'''
date: 2020-03-18


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

data_clean=__import__('02_data_clean')
# from data_clean import *
# print(data_clean)

# 默认词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MIN_COUNT = 3


# 将sentence中的word替换成index
def indexesFromSentence(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]

# 填充+行列转置
def zeroPadding(l,fillvalue=PAD_token):
    return list(itertools.zip_longest(*l,fillvalue=fillvalue))

# 记录PAD_token的位置0，其他位置1
def binaryMatrix(l,value=PAD_token):
    m = []
    for i,seq in enumerate(l):
        m.append([])

        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)

    return m


# 返回填充前的长度和填充后的输入序列张量
def inputVar(l,voc):
    indexes_batch = [indexesFromSentence(voc,sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)

    return padVar,lengths

# 返回填充前的做大长度 和 填充后的输入序列张量，填充后的标记 mask
# l是一个batch的数据
def outputVar(l,voc):
    indexes_batch = [indexesFromSentence(voc,sentence) for sentence in l]

    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)

    return padVar,mask, max_target_len

# 返回一个batch
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")),reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch,voc)

    output, mask, max_target_len = outputVar(output_batch,voc)
    return inp,lengths,output, mask, max_target_len



if __name__ == "__main__":
    save_dir = os.path.join("data","save")
    corpus = "cornell movie-dialogs corpus"
    corpus_name = "cornell movie-dialogs corpus"
    data_file = os.path.join("data",corpus_name,"formatted_movie_lines.txt")
    voc, pairs = data_clean.loadPrepareData(corpus, corpus_name, data_file,save_dir)

    print("\n pairs: ")
    for pair in pairs[:10]:
        print(pair)

    # 修剪voc和对
    pairs = data_clean.trimRareWords(voc, pairs, MIN_COUNT)



    small_batch_size = 5
    batches = batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable: ", input_variable)
    print("lengths: ", lengths)
    print("mask: ", mask)
    print("target_variable: ", target_variable)
    print("max_target_len: ", max_target_len)

    #(max_length,batch_szie)