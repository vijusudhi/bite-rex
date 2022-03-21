import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm as tq
import threading
import pickle
import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

from util import util

class Retrieval():
    def __init__(self, corpus, model, model_type, load_vectors=True):
        self.corpus = corpus
        self.model = model
        self.model_type = model_type  # static or contextual
        
        self.path = '../../data/word2vec/%s/'%self.corpus.name
        self.doc_vectors_path = self.path + 'doc_vectors/%s/' %(self.model_type)
        
        if load_vectors:
            self.load_doc_vectors()
        

    def get_sentence_vector(self, sent, fast=None, lang=None):
        if self.model_type=='static':
            # TODO: identify language automatically
            if fast and lang:
                # here sent is idx
                sent_vector = self.model.encode_with_toks(sent, lang=lang)
                return sent_vector
            
            if isinstance(sent, list):
                sent_vector = []
                for s in sent:
                    sent_vector.append(self.model.encode(s, lang=lang))
                sent_vector = torch.stack(sent_vector)
            else:
                sent_vector = self.model.encode(sent, lang=lang)
        else:
            sent_vector = self.model.encode(sent, show_progress_bar=False)
            sent_vector = torch.from_numpy(sent_vector)
            sent_vector = sent_vector
        return sent_vector
    
    def pre_compute_doc_vectors(self, range_docs):
        logging.info('Computing document vectors')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        logging.info('Running on %s' %device)
        
        NUM_OF_THREADS = 5
        
        def _compute_doc_vector(docs, vectors):
            inc = int(len(docs)/NUM_OF_THREADS)
            for i in range(NUM_OF_THREADS):
                start = i*inc
                end = i*inc+inc
                threading.Thread(target=_thread_function, 
                                 args=(docs, start, end, vectors)
                                ).start()
                
        def _thread_function(docs, start, end, vectors):    
            logging.info("Thread starting with (%d:%d)" %(start, end))        
            for idx in tq(range(start, end)):
                vectors[idx] = self.get_sentence_vector(sent=idx, fast=True)
            logging.info("Thread finishing with (%d:%d)" %(start, end)) 
        
        en_doc_vectors_ = {}
        _compute_doc_vector(docs=self.corpus.corpus_en[range_docs[0]:range_docs[1]], 
                            vectors=en_doc_vectors_)
        self.en_doc_vectors = [vectors[idx] 
                          for idx in range(range_docs[0], range_docs[1]) 
                          if idx in vectors.keys()
                         ]
        
        # de_doc_vectors_ = {}
        # _compute_doc_vector(docs=self.corpus.corpus_de[range_docs[0]:range_docs[1]])
        # self.de_doc_vectors = [vectors[idx] 
        #                   for idx in range(len(self.corpus.corpus_de)) 
        #                   if idx in vectors.keys()
        #                  ]
                
        # util.compress_pickle(self.path+'doc_vectors/en_doc_vectors', self.en_doc_vectors)
        # util.compress_pickle(self.path+'doc_vectors/de_doc_vectors', self.de_doc_vectors)
        # logging.info('Saved document vectors to path: %s'%(self.path+'doc_vectors'))
        
        
    def pre_compute_doc_vectors_v2(self, range_docs):
        logging.info('Computing document vectors')
        
        def get_vectors(range_docs, lang):
            corpus = self.corpus.corpus_en if lang=='en' else self.corpus.corpus_de
            vectors = {}
            for idx in tq(range(range_docs[0], range_docs[1])):
                if self.model_type=='static':
                    vectors[idx] = self.get_sentence_vector(sent=idx, fast=True, lang=lang)
                else:
                    vectors[idx] = self.get_sentence_vector(sent=corpus[idx])
            return vectors
        
        en_dv_dic = get_vectors(range_docs, lang='en')
        de_dv_dic = get_vectors(range_docs, lang='de')
        
        self.en_doc_vectors = torch.stack(list(en_dv_dic.values()))
        self.de_doc_vectors = torch.stack(list(de_dv_dic.values()))
        
        self.en_doc_idx = list(en_dv_dic.keys())
        self.de_doc_idx = list(de_dv_dic.keys())
        
        logging.info('Saving document vectors to path: %s' %(self.doc_vectors_path))
        util.compress_pickle(self.doc_vectors_path+'en_doc_vectors', self.en_doc_vectors)
        util.compress_pickle(self.doc_vectors_path+'de_doc_vectors', self.de_doc_vectors)
        
    def load_doc_vectors(self):
        logging.info('Loading document vectors from path: %s' %(self.doc_vectors_path))
        self.en_doc_vectors = util.decompress_pickle(self.doc_vectors_path+'en_doc_vectors')
        self.de_doc_vectors = util.decompress_pickle(self.doc_vectors_path+'de_doc_vectors')
        

    def get_retrieved_docs(self, query, num_ret, lang=None):
        logging.info('Retrieving top %d results for query: %s' %(num_ret, query))
        query_vector = self.get_sentence_vector(sent=query, lang=lang)
        
        output = F.cosine_similarity(query_vector.unsqueeze(0), self.de_doc_vectors, dim=1)
        ret_ids_de = torch.topk(output, num_ret).indices.cpu().numpy()

        output = F.cosine_similarity(query_vector.unsqueeze(0), self.en_doc_vectors, dim=1)
        ret_ids_en = torch.topk(output, num_ret).indices.cpu().numpy()
        
        try:
            # here indices are not as same as corpus indices
            trans_ret_ids_de = [self.de_doc_idx[idx] for idx in ret_ids_de]
            trans_ret_ids_en = [self.en_doc_idx[idx] for idx in ret_ids_en]
        except:
            # here indices are same as corpus indices
            trans_ret_ids_de = ret_ids_de
            trans_ret_ids_en = ret_ids_en     
        
        ret_docs_de = [self.corpus.corpus_de[idx] for idx in trans_ret_ids_de]
        ret_docs_en = [self.corpus.corpus_en[idx] for idx in trans_ret_ids_en]
        
        ret = {
                'en_doc_ids': trans_ret_ids_en,
                'de_doc_ids': trans_ret_ids_de,
                'en_docs': ret_docs_en,
                'de_docs': ret_docs_de,
            }
        
        return ret

    