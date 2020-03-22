
'''
date:2020-03-18

训练代码:

1.通过编码器前向计算整个批次输入。
2.将解码器输入初始化为SOS_token，将隐藏状态初始化为编码器的最终隐藏状态。
3.通过解码器一次一步地前向计算输入一批序列。
4.如果是 teacher forcing 算法：将下一个解码器输入设置为当前目标;如果是 no teacher forcing 算法：将下一个解码器输入设置为当前解码器输出。
5.计算并累积损失。
6.执行反向传播。
7.裁剪梯度。
8.更新编码器和解码器模型参数。
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
import warnings
warnings.filterwarnings('ignore')


MAX_LENGTH = 10

# 1次迭代
def train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,
    encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=MAX_LENGTH):
    
    # 零化梯度
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 设置设备
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # 正向传递编码器
    encoder_outputs,encoder_hidden = encoder(input_variable,lengths)

    # 创建初始解码器输入（从每个句子的SOS令牌开始)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 将初始解码器隐藏状态设置为编码器的最终隐藏状态
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 确定是否此次迭代使用 teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 通过解码器一次一步的转发一批序列
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,encoder_outputs)

            # Teacher forcing: 下一个输入是当前目标
            decoder_input = target_variable[t].view(1,-1)
            # 计算并累计损失
            mask_loss, nTotal  = maskNLLoss(decoder_output,target_variable[t],mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,encoder_outputs)
            # No teacher forcing 下一个输入是解码器自己的当前输出
            _,topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # 计算并累计损失
            mask_loss, nTotal = maskNLLoss(decoder_output,tarhet_variable[t],mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 直行反向传播
    loss.backward()

    # 梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)

    # 调整模型权重

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# n次迭代
def trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,
    embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,
    clip,save_every,corpus_name,loadFilename):

    # 每次迭代加载batches

    training_batches = [data_ready.batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

    # 初始化
    print("Initializing...")
    start_iteration = 1
    print_loss = 0

    # 为了从断点开始训练
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # 训练循环
    print("Training...")
    pbar = tqdm(range(start_iteration,n_iteration + 1))
    for iteration in pbar:

        training_batch = training_batches[iteration - 1]

        # 从batch中提取字段
        input_variable, lengths,target_variable,mask,max_target_len = training_batch
        # 使用batch运行训练迭代
        loss = train(input_variable,lengths,target_variable,mask,max_target_len,encoder,
            decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip)

        print_loss += loss

        # 打印进度
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("[INFO] Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,iteration,n_iteration*100,print_loss_avg))
            print_loss = 0
        # 保存checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir,model_name,corpus_name,"{}-{}-{}".format(encoder_n_layers,decoder_n_layers,hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                "iteration":iteration,
                "en":encoder.state_dict(),
                "de":decoder.state_dict(),
                "en_opt":encoder_optimizer.state_dict(),
                "de_opt":decoder_optimizer.state_dict(),
                "loss":loss,
                "voc_dict":voc.__dict__,
                "embedding":embedding.state_dict()

                },os.path.join(directory,"{}_{}.tar".format(iteration,'checkpoint')))
        pbar.set_description("聊天机器人训练: Iter:{}/{},Loss:{}".format(iteration,n_iteration,print_loss))
    


if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # ----------------data-----------------------------------

    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    MIN_COUNT = 3

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


    # --------------train---------------------------------
    # 配置训练/优化
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 4000
    print_every = 50
    save_every = 1000

    # 确保dropout layers在训练模型中
    encoder.train()
    decoder.train()

    # 初始化优化器
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # 运行训练迭代
    print("Starting Training!")


    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)