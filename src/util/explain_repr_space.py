from transformers import BertTokenizer
import torch
import re
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from util import projection, split_compounds
import plotly.graph_objects as go
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import GermanStemmer

# define lemmatizer for EN and DE
lemmatizer_en = WordNetLemmatizer()
# lemmatizer_de = Cistem()
lemmatizer_de = GermanStemmer()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=True)

class ExplainReprSpace:
    def __init__(self, query, query_lang, doc, en_ret_docs, de_ret_docs, scraped_en_docs, scraped_de_docs, model, st_encoding):
        self.query = query
        self.doc = doc
        self.model = model.to('cpu')
        self.st_encoding = st_encoding
        self.query_lang = query_lang

        scraped_docs = scraped_en_docs if self.query_lang == 'en' else scraped_de_docs
        self.ret_docs = en_ret_docs if self.query_lang == 'en' else de_ret_docs
        self.ret_docs_all = en_ret_docs + de_ret_docs

        self.query_vec = self.get_sent_vector_contextual(self.query)   

        try:
            self.get_repr_space(scraped_docs, self.ret_docs)
        except:
            self.repr_space = -1
        # self.get_repr_space(scraped_docs, self.ret_docs)


    def get_sent_vector_contextual(self, sent):
        # with torch.no_grad():
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

    
    def get_imp_word_cont(self):
        if self.query_lang == 'de':
            words = []
            for word in self.query.split(" "):
                words.extend(split_compounds.split_compound(word))
            words = [split_compounds.remove_accents(word) for word in words]
        else:
            words = [lemmatizer_en.lemmatize(word) for word in self.query.split(" ")]

        for sp in self.query.split(" "):
            if sp not in words:
                words.append(sp)

        query_vec = self.get_sent_vector_contextual(sent=[self.query])
        word_import = {}
        for word in words:
            word_vec = self.get_sent_vector_contextual(sent=[word])
            word_import[word] = cosine_similarity(word_vec, query_vec)[0]
        word_import = self.sort_dict(word_import)
        imp_word = list(word_import.keys())[-1]
        return imp_word.lower()

    @staticmethod
    def strip_text(doc, word):
        sents = sent_tokenize(doc)
        sent = [s for s in sents if word in s.lower()][0]
        return sent
        

    def plot_umap_intra_sim(self, proj_2d, ret_docs, other_docs):                
        ret_colors = []
        ret_docs_list = []
        ret_projs = []
        ret_sizes = []

        query_proj = proj_2d[:1]
        ret_docs_proj = proj_2d[1:len(ret_docs)+1]
        other_docs_proj = proj_2d[len(ret_docs)+1:]

        cont_query_vec = self.query_vec.reshape([1,-1])
        cont_sim_ret_docs = []
        for vec in self.ret_docs_vecs:
            vec = vec.reshape([1,-1])
            sim = cosine_similarity(cont_query_vec, Y=vec, dense_output=True)[0]
            cont_sim_ret_docs.append(sim[0])

        cont_sim_other_docs = []
        for vec in self.other_docs_vecs:
            vec = vec.reshape([1,-1])
            sim = cosine_similarity(cont_query_vec, Y=vec, dense_output=True)[0]
            cont_sim_other_docs.append(sim[0])   

        # cont_sim_ret_docs = np.asarray(cont_sim_ret_docs)
        # cont_sim_ret_docs = cont_sim_ret_docs/np.max(cont_sim_ret_docs)

        # cont_sim_other_docs = np.asarray(cont_sim_other_docs)
        # cont_sim_other_docs = cont_sim_other_docs/np.max(cont_sim_other_docs)               
                
        for doc, proj, sim in zip(ret_docs, ret_docs_proj, cont_sim_ret_docs):
            if doc == self.doc:
                ret_colors.append('brown')
            else:
                ret_colors.append('red')
            ret_docs_list.append(doc)
            ret_projs.append(proj)
            ret_sizes.append(sim*100)
            
        for doc, proj, sim in zip(other_docs, other_docs_proj, cont_sim_other_docs):
            ret_colors.append('blue')
            stripped_doc = self.strip_text(doc, self.imp_word)
            ret_docs_list.append(stripped_doc)
            ret_projs.append(proj)
            ret_sizes.append(sim*100)
                
        ret_docs_list.append(self.query)
        ret_colors.append('green')
        ret_projs.extend(query_proj)
        ret_sizes.append(100)
        
        return projection.plot_umap_2d(ret_docs_list, ret_projs, ret_colors, ret_sizes)


    def get_repr_space(self, scraped_docs, ret_docs):
        # get the most important word in the query
        self.imp_word_full = self.get_imp_word_cont()
        self.imp_word = self.imp_word_full
        word_vec = self.get_sent_vector_contextual(sent=[self.imp_word])

        # find docs with the impt word
        try:
            docs_with_word_idx = [idx for idx, doc in enumerate(scraped_docs) if re.findall(r'\b%s\b'%self.imp_word, doc.lower())][:100]
            docs_with_word = [scraped_docs[idx] for idx in docs_with_word_idx]
            docs_with_word_vec = self.get_sent_vector_contextual(docs_with_word)
        except:
            lemmatizer = lemmatizer_en.lemmatize if self.query_lang=='en' else lemmatizer_de.stem
            self.imp_word = lemmatizer(self.imp_word)
            docs_with_word_idx = [idx for idx, doc in enumerate(scraped_docs) if re.findall(r'%s'%self.imp_word, doc.lower())][:100]
            docs_with_word = [scraped_docs[idx] for idx in docs_with_word_idx]
            docs_with_word_vec = self.get_sent_vector_contextual(docs_with_word)

        sim = cosine_similarity(word_vec, Y=docs_with_word_vec, dense_output=True)[0]
        ind = np.argsort(sim)
        idxs = [docs_with_word_idx[idx] for idx in ind]
        other_docs = [scraped_docs[idx] for idx in idxs if self.query not in scraped_docs[idx]]
        
        other_docs_filt = []
        for doc in other_docs:
            count = 0
            for query_split in self.query.split(" "):
                 if query_split.lower() in doc.lower():
                     if not self.imp_word in query_split.lower():
                        count += 1
            if count == 0 and len(doc.split(" "))>=2:
                other_docs_filt.append(doc)

        self.other_docs = other_docs_filt[:5]

        self.ret_docs_vecs = self.get_sent_vector_contextual(self.ret_docs_all)
        self.other_docs_vecs = self.get_sent_vector_contextual(self.other_docs)

        vectors = []
        vectors.extend(self.query_vec)
        vectors.extend(self.ret_docs_vecs)
        vectors.extend(self.other_docs_vecs)
        vectors = np.asarray(vectors)

        proj_2d = projection.get_projection_2d(vectors)
        self.repr_space = self.plot_umap_intra_sim(proj_2d, self.ret_docs_all, self.other_docs)

    def get_differences_st_cont(self):
        st_query_vec = self.st_encoding.encode(sent=[self.query], lang=self.query_lang)
        st_sim = []
        for doc in self.other_docs:
            doc_vec = self.st_encoding.encode(sent=[doc], lang=self.query_lang)
            sim = cosine_similarity(st_query_vec, Y=doc_vec, dense_output=True)[0]
            st_sim.append(sim[0])

        cont_query_vec = self.query_vec.reshape([1,-1])
        cont_sim = []
        for vec in self.other_docs_vecs:
            vec = vec.reshape([1,-1])
            sim = cosine_similarity(cont_query_vec, Y=vec, dense_output=True)[0]
            cont_sim.append(sim[0])

        differences = []
        for ind, doc in enumerate(self.other_docs):
            differences.append(
                {
                    'doc': self.strip_text(doc, self.imp_word),
                    'st_sim': "{:.2f}".format(st_sim[ind]),
                    'cont_sim': "{:.2f}".format(cont_sim[ind])
                }
            )
        differences_df = pd.DataFrame(differences)
        return differences_df

    def plot_differences_df(self):
        df = self.get_differences_st_cont()
        fig = go.Figure(data=[go.Table(
            columnwidth = [100,10,10],
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df['doc'], df['st_sim'], df['cont_sim']],
                    fill_color='lavender',
                    align='left'))
        ]) 
        return fig
        