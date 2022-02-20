import torch
from transformers import pipeline
from nltk.corpus import stopwords
import string
import re
import numpy as np
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from util import vectorize
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=True)


class SuggestQueries():
    def __init__(self, model):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=True)
        self.unmasker = pipeline('fill-mask', 
                                 model=self.model.to('cpu'), 
                                 tokenizer=self.tokenizer, top_k=25)
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_de = set(stopwords.words('german'))
        self.punct = list(string.punctuation) 
        sent_trx = SentenceTransformer('/home/sudhi/thesis/thesis_cltr_app/data/models/cont_model')
        self.sent_trx_encoding = vectorize.Encoding(model=[sent_trx], model_type="sentence_bert")

    @staticmethod
    def sort_dict(dict_to_sort):
        return dict(sorted(dict_to_sort.items(), key=lambda item: item[1]))

    # def get_imp_word(self, query):
    #     query_vec = self.sent_trx_encoding.encode(sent=[query])
    #     word_import = {}
    #     for word in query.split(" "):
    #         word_vec = self.sent_trx_encoding.encode(sent=[word])
    #         word_import[word] = cosine_similarity(word_vec, query_vec)[0]
    #     word_import = self.sort_dict(word_import)
    #     imp_word = list(word_import.keys())[-1]
    #     return imp_word 

    def get_imp_word(self, query):
        query_vec = self.get_sent_vector_contextual(sent=[query])
        word_import = {}
        for word in query.split(" "):
            word_vec = self.get_sent_vector_contextual(sent=[word])
            word_import[word] = cosine_similarity(word_vec, query_vec)[0]
        word_import = self.sort_dict(word_import)
        imp_word = list(word_import.keys())[-1]
        return imp_word 


    def get_sent_vector_contextual(self, sent):
        inputs = tokenizer(sent, return_tensors='pt',
                    max_length=512, padding=True, truncation=True)
        # last layer gives the best embeddings
        token_embeddings = self.model(**inputs, output_hidden_states=True)['hidden_states'][-1]
        # Mean Pooling - Take attention mask into account for correct averaging
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sent_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return np.asarray(sent_embedding.detach().numpy())

    @staticmethod
    def sort_dict(dict_to_sort):
        return dict(sorted(dict_to_sort.items(), key=lambda item: item[1]))


    def get_suggestions(self, query):
        imp_word = self.get_imp_word(query)
        masked_query = re.sub(imp_word, "[MASK]", query)
        preds = self.unmasker(masked_query)
        suggestions = []
        for pred in preds:
            token_str = re.sub(" ", "", pred['token_str'])
            if token_str in self.stop_words_en:
                continue
            if token_str in self.stop_words_de:
                continue
            if token_str in self.punct:
                continue
            if re.findall(r'[0-9]', token_str):
                continue        
            if token_str in query.lower().split(" "):
                continue                               
            # TODO: sometimes, comprsiing, inclusing, etc. comes
            # add a rule to avoid these
            # avoid numbers
            suggestion = pred['sequence']
            if suggestion.lower() != query.lower():
                suggestions.append(suggestion)
        return suggestions[:5]