'''
date: 2020-03-17

将语料数据做成[[question,answer],...],
并保存成csv文件

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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join('data',corpus_name)

# 查看预料的前n=10行
def printLines(file,n=10):
    with open(file,'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)



# 将文件中的每一行拆分为字段字典，move_lines.txt
def loadLines(fileName,fields):
    lines = {}
    with open(fileName,'r',encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ") # 'lineID',"characterID","movieID","character","text"
            lineObj = {}  # {"lineID":v1,"characterID":v2,"movieID":v3,"character":v4,"text":v5}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj  # {"L001":{"lineID":v1,"characterID":v2,"movieID":v3,"character":v4,"text":v5},...}
    return lines


# 将loadLines中的行字段分组为基于*movie_conversations.txt*的对话
def loadConversations(fileName,lines,fields):
    conversations = []
    with open(fileName,'r',encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ") # ['characterID','character2ID','movieID','utteranceIDs']
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i] # {'characterID':v1,'character2ID':v2,'movieID':v3,'utteranceIDs':v4}

            lineIds = eval(convObj['utteranceIDs'])  # a string eval to a list,eg:"['L194', 'L195']" to ['L194', 'L195']
            convObj["lines"] = [] # {'characterID':v1,'character2ID':v2,'movieID':v3,'utteranceIDs':v4,"lines":[]}

            for lineId in lineIds:  
                convObj['lines'].append(lines[lineId]) #[{"lineID":v1,"characterID":v2,"movieID":v3,"character":v4,"text":v5},...]

            conversations.append(convObj) #[# {'characterID':v1,'character2ID':v2,'movieID':v3,'utteranceIDs':v4,"lines":[]},...]
    return conversations

# *从对话中提取一对句子[[text1,text2],[text2,text3],...]
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines'])-1):  # 去掉最后一个lines的值，该值没有answer [text1，text2,text3] ---> [text1,text2],[text2,text3]
            inputLine = conversation["lines"][i]['text'].strip()
            targetLine = conversation['lines'][i+1]['text'].strip()

            # 过滤掉错误样本
            if inputLine and targetLine:
                qa_pairs.append([inputLine,targetLine])  #[[text1,text2],[text2,text3],...]

    return qa_pairs




if __name__ == "__main__":

    # 定义新文件的路径

    datafile = os.path.join(corpus,"formatted_movie_lines.txt")
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter,'unicode_escape'))

    # 初始化行dict, 对话列表和字段ID
    lines = {}
    conversations = []
    # 字段
    MOVIE_LINES_FIELDS = ['lineID',"characterID","movieID","character","text"]
    MOVIE_CONVERSATIONS_FIELDS = ['characterID','character2ID','movieID','utteranceIDs']

    # 加载行和进程对话
    print("\n Processing Corpus ... ")
    lines = loadLines(os.path.join(corpus,"movie_lines.txt"),MOVIE_LINES_FIELDS)
    print("\n Loading Conversations ... ")
    conversations = loadConversations(os.path.join(corpus,"movie_conversations.txt"),lines,MOVIE_CONVERSATIONS_FIELDS)

    # 写入新的csv

    print("\n Writing newly formatted file ... ")
    with open(datafile,'w',encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile,delimiter=delimiter,lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # 打印一个样本的行
    print("\n Sample line from file: ")
    printLines(datafile)



    '''
    Processing Corpus ... 

    Loading Conversations ... 

    Writing newly formatted file ... 

    Sample line from file: 
    b"Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\tWell, I thought we'd start with pronunciation, if that's okay with you.\r\n"
    b"Well, I thought we'd start with pronunciation, if that's okay with you.\tNot the hacking and gagging and spitting part.  Please.\r\n"
    b"Not the hacking and gagging and spitting part.  Please.\tOkay... then how 'bout we try out some French cuisine.  Saturday?  Night?\r\n"
    b"You're asking me out.  That's so cute. What's your name again?\tForget it.\r\n"
    b"No, no, it's my fault -- we didn't have a proper introduction ---\tCameron.\r\n"
    b"Cameron.\tThe thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser.  My sister.  I can't date until she does.\r\n"
    b"The thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser.  My sister.  I can't date until she does.\tSeems like she could get a date easy enough...\r\n"
    b'Why?\tUnsolved mystery.  She used to be really popular when she started high school, then it was just like she got sick of it or something.\r\n'
    b"Unsolved mystery.  She used to be really popular when she started high school, then it was just like she got sick of it or something.\tThat's a shame.\r\n"
    b'Gosh, if only we could find Kat a boyfriend...\tLet me see what I can do.\r\n'
    [Finished in 4.1s]
    '''