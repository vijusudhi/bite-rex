from colour import Color
from util import tokenize, vectorize
import re
import pandas as pd
import numpy as np
import plotly.express as px

def get_token_import(model, query, docs, lang):
    toks = [tokenize.get_tokens(doc, lang) for doc in docs]
    toks_proc = []
    for tok in toks:
        toks_proc.extend(tok)        
    toks_proc = list(set(toks_proc))
    
    ret = vectorize.retrieve_cont(model, query, toks_proc, num_ret=len(toks_proc))
    # ret = retrieval.retrieve(query_vec=query_vec, num_ret=len(toks_proc))    
    tokens_import = {tok: imp for tok, imp in zip(ret['en_docs'], ret['en_sim'])}
    
    return tokens_import


def get_token_vectors(encoding, doc, lang):
    tokens = tokenize.get_tokens(doc, lang)       
    tokens = list(set(tokens))
    vectors = encoding.encode(sent=tokens, lang=lang)
    token_vectors = {token: vector for token, vector in zip(tokens, vectors)}
    
    return token_vectors


def get_token_import_doc(model, query, doc, lang):       
    tokens = tokenize.get_tokens(doc, lang)
            
    tok_imp = {}
    for tok in tokens:       
        doc_wo_tok = re.sub(tok, '', doc.lower())
        
        # sent_with_tok_vec = encoding.encode(sent=[doc], lang=lang)
        # sent_wo_tok_vec = encoding.encode(sent=[doc_wo_tok], lang=lang)
        
        # retrieval = vectorize.Retrieval(vectors=[sent_with_tok_vec, sent_wo_tok_vec],
        #                         docs=[[doc], [doc_wo_tok]]
        #                        )
        # ret = retrieval.retrieve(query_vec=query_vec, num_ret=len(doc))
        ret = vectorize.retrieve_cont(model, query, [[doc], [doc_wo_tok]], num_ret=len(doc))
        ref = ret['en_sim'][0]
        sim = ret['de_sim'][0]        
        imp = (ref-sim)/ref
        
        if imp != 0:
            tok_imp[tok] = imp 
            
    tokens_import = dict(sorted(tok_imp.items(), key=lambda item: item[1]))
    
    return tokens_import


def plot_import_bar(tokens_import_en, tokens_import_de=None, use_neg=False):
    tokens_import = {}
    for tok in tokens_import_en:
        tokens_import[tok] = tokens_import_en[tok]
        
    if tokens_import_de:
        for tok in tokens_import_de:
            tokens_import[tok] = tokens_import_de[tok]

    tokens_import = {k: v for k, v in sorted(tokens_import.items(), 
                                             key=lambda item: item[1])
                    }
    
    df = []
    for token in tokens_import:
        importance = tokens_import[token]
        df.append(
            {
                'token': token,
                'importance': importance
            }
        )
    df = pd.DataFrame(df)
    
    if not use_neg:
        classes = [0] * len(tokens_import.values())
        for ind, i in enumerate(tokens_import.values()):
            i = int(i*100)
            idx = (i // 10)
            classes[ind] = idx

        colors = ['%s'%COLORS[clss] for clss in classes]
        height = len(tokens_import)*15
    else:
        colors = []
        for val in tokens_import.values():
            if val >= 0:
                colors.append('green')
            else:
                colors.append('red')
        height = 100
    
    fig = px.bar(df, x="importance", y="token", orientation='h', width=800, height=400)
    fig.update_traces(marker_color=colors) 
    fig.add_vrect(x0=0, x1=0)
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    return fig


def get_query_word_relations(model, query, doc, query_lang='en', doc_lang='en'):
    split_imp = []
    sp = query.split(' ')
    for query_new in sp:
        # query_vector = encoding.encode(sent=[query_new], lang=query_lang)

        # if not np.any(query_vector):
        #    continue

        tokens_import = get_token_import_doc(model=model, 
                                                     query=query_new, 
                                                     doc=doc,
                                                     lang=doc_lang
                                                    )

        txt = get_display_text(doc, tokens_import, mode='background')
        
        
        split_imp.append(
            {
                'split': query_new,
                'text': txt
            }
        )
        
    return split_imp