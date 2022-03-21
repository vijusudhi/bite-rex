from tqdm.notebook import tqdm as tq
import torch
import numpy as np
import datetime
import shutil
import os
from gensim.models import Word2Vec as W2V
import re

import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

from util import util, tokenize

class Word2Vec():
    def __init__(self, 
                 name, 
                 corpus=None, 
                 algn_dic=None, 
                 rerun_data=False, 
                 rerun_train=False, 
                 remove_underscores=False):
        self.name = name
        self.corpus = corpus
        self.algn_dic = algn_dic
        self.aligned_sentences = []
        
        self.path = '../../data/word2vec/%s/'%self.name
        logging.info('Fetching contents from path %s' %self.path)
        
        # if rerun_train:
        #     if rerun_data:
        #         self.generate_train_data()
        #     else:
        #         self.load_train_data()
        #         
        #     if remove_underscores:
        #         self.remove_underscores_from_train_data()
        #         
        #     self.train_model()
        # else:
        #     # self.load_train_data()
        #     self.load_model()
        #     
        # self.build_wv_vocab()

        
    def generate_train_data(self):
        logging.info('Generating train data')
        
        def _generate(en_to_de=True):
            # set the first and second language. this amtters while checking translations
            if en_to_de:
                all_toks_x = self.corpus.all_toks_en
                all_toks_y = self.corpus.all_toks_de
            else:
                all_toks_x = self.corpus.all_toks_de
                all_toks_y = self.corpus.all_toks_en
        
            aligned_sentences = []
            for token_x, token_y in zip(tq(all_toks_x), 
                                        all_toks_y):

                idx_x = [self.corpus.word2idx[word] for word in token_x]
                idx_y = [self.corpus.word2idx[word] for word in token_y]


                # trans_idxes_x = [self.algn_dic.idx2transidx[idx] for idx in idx_x \
                #                  if idx in self.algn_dic.idx2transidx.keys()]
                trans_idxes_x = []
                for idx in idx_x:
                    if idx in self.algn_dic.idx2transidx.keys():
                        trans_idxes_x.append(self.algn_dic.idx2transidx[idx])
                    else:
                        trans_idxes_x.append(-1)

                # print(trans_idxes_x)

                # TODO: consider multiple trans indices
                # trans_idx_x = [-1] * len(idx_x)
                # for ind, _ in enumerate(trans_idxes_x):
                #     for idx in _:
                #         if idx in idx_y:
                #             trans_idx_x[ind] = idx
                trans_idx_x = trans_idxes_x

                all_trans_idxes = []
                for ind, idx in enumerate(idx_x):
                    if trans_idx_x[ind] != -1:
                        all_trans_idxes.append(idx_x[:ind]+[trans_idx_x[ind]]+idx_x[ind+1:])

                for idxes in all_trans_idxes:
                    tokens = [self.corpus.idx2word[idx] for idx in idxes]
                    aligned_sentences.append(tokens)
                
            return aligned_sentences
                
        self.aligned_sentences = []
        # get EN - DE aligned sentences. here main sentences are in EN with DE replacements
        self.aligned_sentences.extend(_generate(en_to_de=True))
        # get DE - EN aligned sentences. here main sentences are in DE with EN replacements
        self.aligned_sentences.extend(_generate(en_to_de=False))
        
        util.compress_pickle(self.path+'train/aligned_sentences', self.aligned_sentences)
        logging.info('Saved train data to path: %s'%(self.path+'train/aligned_sentences'))        
        
    def load_train_data(self):
        logging.info('Loading train data from path: %s' %(self.path+'train/aligned_sentences'))
        self.aligned_sentences = util.decompress_pickle(self.path+'train/aligned_sentences')
        
    def remove_underscores_from_train_data(self):
        cleaned_sentences = []
        for tokens in tq(self.aligned_sentences):
            cleaned_tokens = []
            for token in tokens:
                cleaned_tokens.append(re.sub('_', ' ', token))
            cleaned_sentences.append(cleaned_tokens)
        self.aligned_sentences = cleaned_sentences
        
        util.compress_pickle(self.path+'train/aligned_sentences', self.aligned_sentences)
        logging.info('Saved train data to path: %s'%(self.path+'train/aligned_sentences'))        
        
    def train_model(self, vector_size=300, window=10, min_count=10):
        logging.info('Training Word2Vec model')
        self.model = W2V(sentences=self.aligned_sentences, 
                        vector_size=300, 
                        window=10, 
                        min_count=10, 
                        workers=10,
                        sg=1,
                        compute_loss=True)
        now = datetime.datetime.today() 
        ts = now.strftime('%d-%m-%Y-%H-%M')
        dest = os.path.join(self.path+"models/static/"+ts)
        if not os.path.exists(dest):
            os.makedirs(dest) # creat dest dir        
        self.model.save(self.path+"models/static/"+ts+"/word2vec.model")
        logging.info('Saved Word2Vec model to path: %s'%(self.path+"models/static/"+ts+"/word2vec.model"))
        
    def load_model(self):
        for x in os.walk(self.path+"models/static/"):
            _, folders, _ = x
            break        
        now = datetime.datetime.today()
        diff = []
        for folder in folders:
            x = datetime.datetime.strptime(folder, '%d-%m-%Y-%H-%M')
            diff.append(now- x)
        ts = folders[diff.index(min(diff))]
        logging.info('Loading Word2Vec model from path: %s' %(self.path+"models/static/"+ts+"/word2vec.model"))
        self.model = W2V.load(self.path+"models/static/"+ts+"/word2vec.model")
        
        
    def build_wv_vocab(self):
        logging.info('Building Word2Vec vocab')
        wv_vocab = []
        for idx in range(len(self.model.wv)):
            wv_vocab.append(self.model.wv.index_to_key[idx])
        self.wv_vocab = wv_vocab
        

    def encode(self, sent, lang):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # TODO: vector dimension is hardcoded. Change that.
        sent_vector = torch.zeros([300], dtype=torch.float32, device=device)
        
        tokens = tokenize.get_tokens(sent, lang=lang)  # TODO: insert language detection. 
        # think about query. user shall not give the query language. the system should
        # identify it.

        for token in tokens:
            if token not in self.wv_vocab:
                # count -= 1
                continue
            # Get IN embeddings for queries and OUT embeddings for documents 
            # vec = model.wv[token] if lang=='en' else model.syn1neg[wv_vocab.index(token)]
            vec = self.model.syn1neg[self.wv_vocab.index(token)] + self.model.wv[token]
            # vec = model.wv[token] 
            # word_vector =  torch.mul(torch.Tensor(np.asarray(vec), device=device), word_idf[token])
            word_vector = torch.Tensor(np.asarray(vec), device=device)
            # Max normalization
            # word_vector = torch.div(word_vector, torch.linalg.norm(word_vector))
            sent_vector = torch.add(sent_vector, word_vector)

        # Max normalization
        # sent_vector = torch.div(sent_vector, torch.max(sent_vector))
        # Mean of sentence vector
        # sent_vector = torch.div(sent_vector, count)

        return sent_vector.squeeze(0)     
    
    
    def encode_with_toks(self, idx, lang):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # TODO: vector dimension is hardcoded. Change that.
        sent_vector = torch.zeros([300], dtype=torch.float32, device=device)
        
        tokens = self.corpus.all_toks_en[idx] if lang=='en' else self.corpus.all_toks_de[idx]

        for token in tokens:
            try:
                tok_vec = self.model.wv[token]
                tok_neg_vec = self.model.syn1neg[self.wv_vocab.index(token)]
            except:
                tok_vec = np.zeros(300, dtype=np.float32)
                tok_neg_vec = np.zeros(300, dtype=np.float32)
                
            vec = np.add(tok_vec, tok_neg_vec)
            word_vector = torch.Tensor(np.asarray(vec), device=device)
            sent_vector = torch.add(sent_vector, word_vector)

        return sent_vector.squeeze(0)      