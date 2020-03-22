'''
date: 2020-03-18

测试：使用贪婪解码
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
from tqdm import tqdm
data_clean=__import__('02_data_clean')
data_ready=__import__('03_data_ready')
from model import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# ----------------data-----------------------------------

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MIN_COUNT = 3

MAX_LENGTH = 10

class GreedySearchDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(GreedySearchDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self,input_seq,input_length,max_length):
        # 通过编码器模型转发输入
        encoder_outputs,encoder_hidden = self.encoder(input_seq,input_length)
        # 准备编码器的最终隐藏层作为解码器的第一个隐层输入
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # 使用SOS_token初始化解码输入
        decoder_input  = torch.ones(1,1,device=device,dtype=torch.long)*SOS_token
        # 初始化张量以将解码后的单词附加到
        all_tokens = torch.zeros([0],device=device,dtype=torch.long)
        all_scores = torch.zeros([0],device=device)

        # 一次迭代的解码一个词tokens
        for _ in range(max_length):
            # 正向通过解码器
            decoder_output,decoder_input = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            # 获得最可能的单词标记及其softmax得分
            decoder_scores,decoder_input = torch.max(decoder_output,dim=1)
            # 记录token和分数
            all_tokens = torch.cat((all_tokens,decoder_input),dim=0)
            all_scores = torch.cat((all_scores,decoder_scores),dim=0)

            # 准备当前令牌作为下一个解码器输入（添加维度）
            decoder_input = torch.unsqueeze(decoder_input,0)
        # 返回收集到的词tokens和分数
        return all_tokens,all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### 格式化输入句子作为batch
    # words -> indexes
    indexes_batch = [data_ready.indexesFromSentence(voc, sentence)]
    # 创建lengths张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置batch的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用合适的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用searcher解码句子
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 获取输入句子
            input_sentence = input('> ')
            # 检查是否退出
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 规范化句子
            input_sentence = data_clean.normalizeString(input_sentence)
            # 评估句子
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 格式化和打印回复句
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


if __name__ == "__main__":
    # 将dropout layers设置为eval模式 
    save_dir = os.path.join("data","save")
    corpus = "cornell movie-dialogs corpus"
    corpus_name = "cornell movie-dialogs corpus"
    data_file = os.path.join("data",corpus_name,"formatted_movie_lines.txt")
    voc, pairs = data_clean.loadPrepareData(corpus, corpus_name, data_file,save_dir)

    # print("\n pairs: ")
    # for pair in pairs[:10]:
    #     print(pair)

    # 修剪voc和对
    pairs = data_clean.trimRareWords(voc, pairs, MIN_COUNT)

    small_batch_size = 5
    batches = data_ready.batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    # print("input_variable: ", input_variable)
    # print("lengths: ", lengths)
    # print("mask: ", mask)
    # print("target_variable: ", target_variable)
    # print("max_target_len: ", max_target_len)



    # ----------------配置模型--------------------------------
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # 设置检查点以加载; 如果从头开始，则设置为None
    loadFilename = None
    checkpoint_iter = 4000
    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                               '{}-{}-{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))

    # 如果提供了loadFilename，则加载模型
    if loadFilename:
        # 如果在同一台机器上加载，则对模型进行训练
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # 初始化词向量
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # 初始化编码器 & 解码器模型
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # 使用合适的设备
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    encoder.eval()
    decoder.eval()

    # 初始化探索模块
    searcher = GreedySearchDecoder(encoder, decoder)

    # 开始聊天（取消注释并运行以下行开始）
    evaluateInput(encoder, decoder, searcher, voc)
