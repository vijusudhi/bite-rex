import numpy as np
import re
# import faiss

from util import tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

from sklearn.metrics.pairwise import cosine_similarity

import sys
import logging
logging.disable(sys.maxsize)

ST_DIM = 300

from tqdm import tqdm

USE_FSE, USE_TRANSFORMERS = True, True

if USE_FSE:
    from fse.models import Average
    from fse import IndexedList

if USE_TRANSFORMERS:
    from transformers import BertTokenizer
    import torch
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

class Encoding:
    def __init__(self, model, model_type='static'):
        self.model = model
        self.model_type = model_type
        
    def encode(self, sent, lang=None):
        if sent[0] == '':
            sent = ['cars']        
        
        if self.model_type == 'static':
            return self.encode_static(sent, lang)
        elif self.model_type == 'sentence_bert':
            return self.encode_contextual(sent)
        elif self.model_type == 'multi_bert':
            return self.encode_contextual_pre(sent)
    
    def encode_static(self, sent, lang):
        def get_doc_train(doc):
            toks = tokenize.get_tokens(doc, lang)
            return toks

        def get_doc_list_and_tuple(docs):
            doc_list = []
            doc_tuple = []
            for ind, doc in enumerate(docs):
                sp = get_doc_train(doc)
                doc_list.append(sp)
                doc_tuple.append((sp, ind))
            return doc_list, doc_tuple

        def get_doc_vectors(model, doc_tuple):
            doc_vectors = []
            for vec in model.infer(doc_tuple):
                doc_vectors.append(vec)
            return doc_vectors
        
        keyed_vec = self.model[0] if lang=='en' else self.model[1]
        model = Average(keyed_vec)
        doc_list, doc_tuple = get_doc_list_and_tuple(sent)
        indexed = IndexedList(doc_list)
        # Try exccept added because FSE expects doc_list with
        # atleast one known token i.e. word in vocab.
        # Otherwise it throws an error
        try:
            model.train(indexed)   
            sent_vector = get_doc_vectors(model, doc_tuple)
            sent_vector = np.asarray(sent_vector)
        except:
            sent_vector = np.zeros(ST_DIM).reshape([1,-1])
            
        return sent_vector

    def encode_post_hoc(self, sent, lang):
        model = self.model[0] if lang=='en' else self.model[1] 
        # vocab = self.vocab[0] if lang=='en' else self.vocab[1]

        tokens = tokenize.get_tokens(sent, lang=lang)
        word_vectors = [np.asarray(model[token]) for token in tokens \
                        if token in model]
        
        if word_vectors:
            sent_vector = np.sum(word_vectors, axis=0).reshape([1,-1])
        else:
            sent_vector = np.zeros(ST_DIM).reshape([1,-1])

        return sent_vector

    def encode_static_comb(self, sent):
        print("sentence", sent)
        sent_split = sent.split(' ')
        sent_vector = np.zeros([300])
        count = 0
        for sp in sent_split:
            if sp.lower() in self.model[0]:
                count += 1
                sent_vector += self.model[0][sp.lower()]
            if sp.lower() in self.model[1]:
                count += 1
                sent_vector += self.model[1][sp.lower()]            
        print("count", count)
        if count != 0:
            sent_vector /= count
        sent_vector = np.asarray(sent_vector).reshape([1,-1])
        print(sent_vector.shape)
        return sent_vector
    
    def encode_contextual(self, sent):
        sent_vector = self.model[0].encode(sent, show_progress_bar=False)
        return sent_vector
    
    def encode_contextual_pre(self, sent):
        print(sent)
        sent_vectors = []
        for s in sent:
            with torch.no_grad():
                inputs = tokenizer(s, return_tensors='pt',
                              max_length=512, truncation=True,
                              padding='max_length').to('cuda')
                hidden_states = self.model[0](**inputs, output_hidden_states=True)['hidden_states']
                token_embeddings = hidden_states[-1] #First element of model_output contains all token embeddings
                # token_embeddings = torch.sum(torch.cat(hidden_states[-1:13]), 0).unsqueeze(0)
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sent_vector = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                                        torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                sent_vectors.append(sent_vector.squeeze(0).detach().cpu())
                         
        sent_vectors = torch.stack(sent_vectors)
        return np.asarray(sent_vectors)
    
    
class Retrieval:
    def __init__(self, 
                 vectors,
                 docs
                ):
        self.num_vectors = len(vectors)
        # en_index = faiss.IndexFlatIP(DIM)
        # en_index = faiss.IndexFlatL2(DIM)
        # en_index.add(np.asarray(vectors[0]))
        # self.en_index = en_index
        self.en_docs = docs[0]
        self.en_vectors = vectors[0]
        
        if self.num_vectors == 2:
            # de_index = faiss.IndexFlatIP(DIM)
            # de_index = faiss.IndexFlatL2(DIM)
            # de_index.add(np.asarray(vectors[1]))            
            # self.de_index = de_index
            self.de_docs = docs[1]  
            self.de_vectors = vectors[1]
        
        
    def retrieve(self, query_vec, num_ret):       
        sim_en = cosine_similarity(query_vec, Y=self.en_vectors, 
                                   dense_output=True)[0]
        ind_en = np.argsort(sim_en)[-num_ret:][::-1]
        
        if self.num_vectors == 2:
            sim_de = cosine_similarity(query_vec, Y=self.de_vectors, 
                                       dense_output=True)[0]
            ind_de = np.argsort(sim_de)[-num_ret:][::-1]            
            
            ret = {
                    'en_docs': [self.en_docs[ind] for ind in ind_en],
                    'en_sim': [sim_en[ind] for ind in ind_en],
                    'en_ind': ind_en,
                    'de_docs': [self.de_docs[ind] for ind in ind_de],
                    'de_sim': [sim_de[ind] for ind in ind_de],
                    'de_ind': ind_de,
                    'en_vectors': [self.en_vectors[ind] for ind in ind_en],
                    'de_vectors': [self.de_vectors[ind] for ind in ind_de],
                  } 
        else:
            ret = {
                    'en_docs': [self.en_docs[ind] for ind in ind_en],
                    'en_sim': [sim_en[ind] for ind in ind_en],
                    'en_ind': ind_en,
                    'sim': sim_en
                  }            
            
        return ret

    def retrieve_with_tfidf(self, query, num_ret):
        X = vectorizer.fit_transform(self.en_docs+[query])
        query_vec = X[-1]
        doc_vectors = X[:-1]
        sim_en = cosine_similarity(query_vec, Y=doc_vectors, 
                           dense_output=True)[0]
        ind_en = np.argsort(sim_en)[-num_ret:][::-1]
        ret = {
                'en_docs': [self.en_docs[ind] for ind in ind_en],
                'en_sim': [sim_en[ind] for ind in ind_en],
                'en_ind': ind_en,
                'sim': sim_en
              }         
        return ret

    def combine_and_retrieve(self, static_weight, threshold, retrieved_static, retrieved_tfidf, num_ret):
        max_static = max(retrieved_static['sim'])
        max_tfidf = max(retrieved_tfidf['sim'])

        max_static = 1 if max_static == 0 else max_static
        max_tfidf = 1 if max_tfidf == 0 else max_tfidf

        sim_static = retrieved_static['sim']/max_static
        sim_tfidf = retrieved_tfidf['sim']/max_tfidf
        sim_en = static_weight * sim_static + (1-static_weight) * sim_tfidf
        sim_en /= max(sim_en)
        ind_en = np.argsort(sim_en)[-num_ret:][::-1]

        ret = {
                'en_docs': [self.en_docs[ind] for ind in ind_en if sim_en[ind] > threshold],
                'en_sim': [sim_en[ind] for ind in ind_en if sim_en[ind] > threshold],
                'en_ind': [ind for ind in ind_en if sim_en[ind] > threshold],
              }         
        return ret        

    
    
def get_preds(model, query, docs):
    preds = []
    for doc in tqdm(docs):
        with torch.no_grad():
            inputs = tokenizer(query, doc, return_tensors='pt',
                          max_length=512, padding=True, truncation=True).to('cuda')
            output = model(**inputs)
            pred = output.seq_relationship_logits.detach().cpu().numpy()[0]
            preds.append(pred)
            
    pred_isnext = [pred[0] for pred in preds]
    return pred_isnext            

def get_top_k(preds, docs, num_ret):
    ind = np.argsort(preds)[-num_ret:][::-1]
    return [docs[i] for i in ind]


def retrieve_cont(model, query, docs, num_ret):
    en_preds = get_preds(model, query, docs[0])
    en_ret_docs = get_top_k(en_preds, docs[0], num_ret)
    
    if len(docs) == 2:
        de_preds = get_preds(model, query, docs[1])
        de_ret_docs = get_top_k(de_preds, docs[1], num_ret)
        ret = {
                'en_docs': en_ret_docs,
                'en_sim': en_preds,
                'de_docs': de_ret_docs,
                'de_sim': de_preds,
              } 
    else:
        ret = {
                'en_docs': en_ret_docs,
                'en_sim': en_preds
              }
    return ret