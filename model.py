'''
date: 2020-03-18

Seq2Seq based on Attention

reference:
    https://arxiv.org/pdf/1406.1078v3.pdf
    https://arxiv.org/abs/1409.0473
    https://arxiv.org/abs/1409.3215
    https://arxiv.org/abs/1508.04025

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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 编码器一个单层的GRU序列
class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):

        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化GRU,input_size和hidden_size都设置为’hidden_size'
        # 因为我们输入大小是一个嵌入了多个特征的单词==hidden_size
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,
            dropout=(0 if n_layers == 1 else dropout),bidirectional=True)

    def forward(self,input_seq,input_lengths,hidden=None):
        # 将单词索引转化为词向量
        embedded = self.embedding(input_seq)
        # 为RNN模块打包填充batch序列
        packed = nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        # 正向通过GRU
        outputs,hidden = self.gru(packed,hidden)
        # 打开填充
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 总和双向GRU输出
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]

        # 返回输出和最终隐藏状态
        return outputs,hidden

# 基于attention的解码器(global attention)

# global attention(Luong的ateention)
class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method = method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,"is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            # hidden_size = A*self.hidden_size + b
            self.attn = torch.nn.Linear(self.hidden_size,hidden_size)
        if self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2,hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_szie))


    def dot_score(self,hidden,encoder_output):
        # hidden 是当前的目标解码器的状态
        return torch.sum(hidden * encoder_output,dim=2)

    def general_score(self,hidden,encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy,dim=2)

    def concat_score(self,hidden,encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v * energy,dim=2)

    def forward(self,hidden,encoder_output):
        # 根据给定的方法计算注意力
        if self.method == "general":
            attn_energies = self.general_score(hidden,encoder_output)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden,encoder_output)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden,encoder_output)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies,dim=1).unsqueeze(1)

# 解码器

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义层
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

        self.attn = Attn(attn_model,hidden_size)

    def forward(self,input_step,last_hidden,encoder_outputs):
        # 注意，我们一次运行一步（一个输出单词）
        # 获取当前输入字的嵌入

        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 通过单向GRU转发
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 从当前GRU输出计算注意力
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量,bmm矩阵乘法
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 使用Luong的公式5 连接加权上下文向量和GRU输出
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # 使用Luong的公式6预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        # 返回输出和在最终隐藏状态
        return output, hidden



# Loss function

def maskNLLoss(inp,target,mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean() # 只考虑mask=1的部分的均值
    loss = loss.to(device)
    return loss,nTotal.item()