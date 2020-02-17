#文本预处理
#步骤：
#1）读入文本
#2）分词
#3）建立字典，将每个词映射到唯一的索引index
#4）将文本从词的序列转换为索引的序列，方便输入模型


import collections
import re

#读入文本
def read_time_machine():
    with open('./Datasets/pride-and-prejudice.txt','r') as f:
        lines=[re.sub('[^a-z]+',' ',line.strip().lower())
              for line in f]
    return lines


lines = read_time_machine()
print('# sentences %d' % len(lines))
# print( lines[0:6])


#分词
def tokenize(sentences,token='word'):
    """Split sentences into word or char tokens"""
    if token=='word':
        return [sentence.split(' ') for sentence in sentences]
    elif token=='char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)

tokens=tokenize(lines)#二维列表


def count_corpus(tokenss):
    tokens=[tk for st in tokenss for tk in st]
    return collections.Counter(tokens)#返回一个字典，记录每个词的出现次数


#建立字典
class Vocab(object):
    def __init__(self,tokens,min_freq=0,use_special_tokens=False):
        counter=count_corpus(tokens)  #<词，词频>
        self.token_freqs=list(counter.items())
        self.idx_to_token=[]
        if use_special_tokens:
            #padding, begin of sentence, end of sentence, unknown
            self.pad,self.bos,self.eos,self.unk=(0,1,2,3)
            self.idx_to_token+=['<pad>','<bos>','<eos>','<unk>']
        else:
            self.unk=0
            self.idx_to_token+=['<unk>']
        self.idx_to_token+=[token for token,freq in self.token_freqs
                            if freq>=min_freq and token not in self.idx_to_token]
        self.token_to_idx=dict()
        for idx,token in enumerate(self.idx_to_token):
            self.token_to_idx[token]=idx


    def __len__(self):
        return len(self.idx_to_token)


    def __getitem__(self, tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]


    def to_tokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens[index] for index in indices]





if __name__=='__main__':
    pass
    # print(tokens[0:2])
    # counter = count_corpus(tokens)
    # token_freqs = list(counter.items())
    # print(token_freqs)
    # vocab = Vocab(tokens)
    # print(list(vocab.token_to_idx.items())[0:10])

    # for i in range(0, 4):
    #     print('words:', tokens[i])
    #     print('indices:', vocab[tokens[i]])




