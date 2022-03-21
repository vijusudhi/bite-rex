import os
import pandas as pd
from tqdm.notebook import tqdm as tq
import threading

import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

from util import util

class AlignmentDic():
    def __init__(self, name, corpus, rerun=False):
        self.name = name  # path of dic dataframe
        self.path = '../../data/word2vec/%s/'%self.name
        assert os.path.exists(self.path+'alg/algn_dic.pbz2'), 'Alignment dictionary not found'
        
        logging.info('Fetching alignment dictionary from path %s' %self.path)
        self.corpus = corpus
        self.rerun = rerun
        
        self.dic = util.decompress_pickle(self.path+'alg/algn_dic')
        # lower case all entries in the dictionary
        self.dic.en = self.dic.en.str.lower()
        self.dic.de = self.dic.de.str.lower()
        # remove entries not in vocab
        self.dic = self.dic[self.dic.de.isin(self.corpus.vocab_de)]
        self.dic = self.dic[self.dic.en.isin(self.corpus.vocab_en)]
        
        # if self.rerun:
        #     self.generate_trans_dic()
        # else:
        #     self.load_trans_dic()        
        
    def get_de_to_en(self, word_de):
        return self.dic[self.dic.de.str.lower() == word_de.lower()].en.to_list()

    def get_en_to_de(self, word_en):
        return self.dic[self.dic.en.str.lower() == word_en.lower()].de.to_list()
    
    def generate_trans_dic_thread(self, en_to_de=True):
        logging.info('Generating translation dictionary')
        idx2transidx = {}
        
        NUM_OF_THREADS = 10
        
        self.idx2transidx = {}
        self.idx2transidx_de = {}
        
        if en_to_de:
            vocab = self.corpus.vocab_en
            get_trans = self.get_en_to_de
            idx2transidx = self.idx2transidx
        else:
            vocab = self.corpus.vocab_de
            get_trans = self.get_de_to_en
            idx2transidx = self.idx2transidx_de
        
        def _thread_function(start, end):
            logging.info("Thread starting with (%d:%d)" %(start, end))
            for word in tq(vocab[start:end]):
                # get the de translation of the input word
                trans_words = get_trans(word)
                # fetch the unique + lower cased translation
                # trans_words = list(set([str(w).lower() for w in trans_words]))
                if trans_words:
                    # TODO: consider multiple trans words, maybe consider probability
                    # trans_word = trans_words[0].lower()
                    trans_idx = self.corpus.word2idx[trans_word]
                else:
                    trans_idx = -1
                # print(trans_words)
                # if the translated word is in the vocabulary (of the other language), add it to list
                # get the indices of the translated words and set this as the dictionary value
                # trans_idx = [self.corpus.word2idx[w] for w in trans_words if w in self.corpus.vocab_de]
                # trans_idx = self.corpus.word2idx[trans_word]
                idx2transidx[self.corpus.word2idx[word]] = trans_idx
            logging.info("Thread finishing with (%d:%d)" %(start, end))
        
        idx2transidx = {}
        inc = int(len(vocab)/NUM_OF_THREADS)
        print('inc', inc)
        # inc = 5
        for i in range(NUM_OF_THREADS):
            start = i*inc
            end = i*inc+inc
            threading.Thread(target=_thread_function, 
                             args=(start, end)
                            ).start()     
            
    
    def generate_trans_dic(self):
        logging.info('Generating translation dictionary')
        
        self.idx2transidx = {}
        for en, de in zip(tq(self.dic.en), 
                          self.dic.de):
            
            en_idx = self.corpus.word2idx[en]
            de_idx = self.corpus.word2idx[de]

            self.idx2transidx[en_idx] = de_idx
            self.idx2transidx[de_idx] = en_idx
                    
        util.compress_pickle(self.path+'alg/transidx', self.idx2transidx)
        logging.info('Saving translation dictionary to path: %s'%(self.path+'alg/transidx'))     
        
    def load_trans_dic(self):
        logging.info('Loading translation dictionary from path: %s'%(self.path+'alg/transidx'))
        self.idx2transidx = util.decompress_pickle(self.path+'alg/transidx')