import torch
import random
import zipfile
import d2lzh_pytorch as d2l


corpus_indices, char_to_idx, idx_to_char, vocab_size=d2l.load_data_jay_lyrics()

sample=corpus_indices[:20]
# print('chars: ',''.join([idx_to_char[idx] for idx in sample]))
# print('indices: ',sample)







if __name__=='__main__':
    my_seq = list(range(30))
    for X, Y in d2l.data_iter_random(my_seq, batch_size=2, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')

    for X, Y in d2l.data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')



