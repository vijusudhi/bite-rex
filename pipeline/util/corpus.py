import os
import pandas as pd
from tqdm.notebook import tqdm as tq
from pathlib import Path
import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

from util import tokenize, util

class Corpus():
    def __init__(self, 
                 name, 
                 doc, 
                 tokenize_corpus=False, 
                 load_vocab=True,
                 load_tokens=False):
        self.name = name # epo, europarl, wiki
        self.doc = doc # doc in df format. should have 'en' and 'de' values
        
        assert 'en' in list(self.doc), 'Dataframe does not have <en> values'
        assert 'de' in list(self.doc), 'Dataframe does not have <de> values'
        
        self.corpus_en = self.doc.en.to_list()
        self.corpus_de = self.doc.de.to_list()

        self.path = '../../data/word2vec/%s/'%self.name
        if tokenize_corpus:
            self.tokenize_corpus()
        if load_vocab:
            self.load_vocab()
        if load_tokens:
            self.load_tokens()
            
        self.word2idx = {word:idx for idx, word in enumerate(self.vocab_en+self.vocab_de)}
        self.idx2word = {idx:word for idx, word in enumerate(self.vocab_en+self.vocab_de)}
        
        logging.info('Artefacts in the path: %s' %self.path)
        logging.info('Sentences\n\tLang\tSize\n\tEN\t%d\n\tDE\t%d' 
                     %(len(self.corpus_en), len(self.corpus_de)))        
        logging.info('Vocab\n\tLang\tSize\n\tEN\t%d\n\tDE\t%d' 
                     %(len(self.vocab_en), len(self.vocab_de)))        
        
    def tokenize_corpus(self):
        def _tokenize(lang):
            # print('Tokenizing for lang: %s...' %lang)
            logging.info('Tokenizing for lang: %s' %lang)
            corpus = self.corpus_en if lang=='en' else self.corpus_de
            vocab = []
            all_tokens = []
            for text in tq(corpus):
                tokens = tokenize.get_tokens(text, lang, lemmatize=False)
                vocab.extend(tokens)
                all_tokens.append(tokens)
            vocab = list(set(vocab))
            return vocab, all_tokens
        
        self.vocab_en, self.all_toks_en = _tokenize(lang='en')
        self.vocab_de, self.all_toks_de = _tokenize(lang='de')
        
        util.compress_pickle(self.path+'vocab/vocab_en', self.vocab_en)
        util.compress_pickle(self.path+'vocab/vocab_de', self.vocab_de)
        util.compress_pickle(self.path+'tokens/all_toks_en', self.all_toks_en)
        util.compress_pickle(self.path+'tokens/all_toks_de', self.all_toks_de)
        
    def load_vocab(self):
        assert os.path.exists(self.path+'vocab/vocab_en.pbz2'), 'Vocab en not found'
        assert os.path.exists(self.path+'vocab/vocab_de.pbz2'), 'Vocab de not found'
        
        logging.info('Loading vocab')
        self.vocab_en = util.decompress_pickle(self.path+'vocab/vocab_en')
        self.vocab_de = util.decompress_pickle(self.path+'vocab/vocab_de')
        
        
    def load_tokens(self):
        assert os.path.exists(self.path+'tokens/all_toks_en.pbz2'), 'All tokens en not found'
        assert os.path.exists(self.path+'tokens/all_toks_de.pbz2'), 'All tokens de not found'
        
        logging.info('Loading tokens')
        self.all_toks_en = util.decompress_pickle(self.path+'tokens/all_toks_en')
        self.all_toks_de = util.decompress_pickle(self.path+'tokens/all_toks_de')        