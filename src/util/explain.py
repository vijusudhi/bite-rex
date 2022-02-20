from colour import Color
from util import tokenize, vectorize
import re
import pandas as pd
import numpy as np
import plotly.express as px

def get_token_import(encoding, query_vec, docs, lang):
    toks = [tokenize.get_tokens(doc, lang) for doc in docs]
    toks_proc = []
    for tok in toks:
        toks_proc.extend(tok)        
    toks_proc = list(set(toks_proc))
    toks_vec = encoding.encode(sent=toks_proc, lang=lang)
    
    retrieval = vectorize.Retrieval(vectors=[toks_vec],
                                    docs=[toks_proc]
                                   )
    ret = retrieval.retrieve(query_vec=query_vec, num_ret=len(toks_proc))    
    tokens_import = {tok: imp for tok, imp in zip(ret['en_docs'], ret['en_sim'])}
    
    return tokens_import


def get_token_vectors(encoding, doc, lang):
    tokens = tokenize.get_tokens(doc, lang)       
    tokens = list(set(tokens))
    vectors = encoding.encode(sent=tokens, lang=lang)
    token_vectors = {token: vector for token, vector in zip(tokens, vectors)}
    
    return token_vectors


def get_token_import_doc(encoding, query_vec, doc, lang):       
    tokens = tokenize.get_tokens(doc, lang)
            
    tok_imp = {}
    for tok in tokens:       
        doc_wo_tok = re.sub(tok, '', doc.lower())
        
        sent_with_tok_vec = encoding.encode(sent=[doc], lang=lang)
        sent_wo_tok_vec = encoding.encode(sent=[doc_wo_tok], lang=lang)
        
        retrieval = vectorize.Retrieval(vectors=[sent_with_tok_vec, sent_wo_tok_vec],
                                docs=[[doc], [doc_wo_tok]]
                               )
        ret = retrieval.retrieve(query_vec=query_vec, num_ret=len(doc))
        ref = ret['en_sim'][0]
        sim = ret['de_sim'][0]        
        imp = (ref-sim)/ref
        
        if imp != 0:
            tok_imp[tok] = imp 
            
    tokens_import = dict(sorted(tok_imp.items(), key=lambda item: item[1]))
    
    return tokens_import    


def get_colors():
    red = Color("#f9ebea")
    colors = list(red.range_to(Color("#d98880"), 11))
    return colors

def get_display_text(doc, tokens_import, mode='background'):    
    def get_color_class(values):
        classes = [0] * len(values)
        for ind, i in enumerate(values):
            i = int(i)
            idx = i // 10
            if idx > 0:
                classes[ind] = idx
        return classes
    
    tok_imp = []
    doc = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', ' ', doc)
    for word in doc.split(' '):
        w_l = word.lower()
        if w_l in tokens_import.keys():
            tok_imp.append(tokens_import[w_l])
        else:
            tok_imp.append(0)
    
    factor = 1 if max(tok_imp) == 0 else max(tok_imp)
    values = (np.asarray(tok_imp)/factor)*100
    COLORS = get_colors()
    classes = get_color_class(values)
        
    doc_str = ''        
    if mode == 'bold':
        for word, clss in zip(doc.split(' '), classes):
            if clss > 9:
                doc_str += '<b>%s</b>' %(word) + ' '
            else:
                doc_str += '%s' %(word) + ' '
    else: 
        mode_str = 'background-color' if mode == 'background' else 'color'
        COLORS[0] = 'white' if mode == 'background' else 'black'
        for word, clss in zip(doc.split(' '), classes):
            doc_str += '<span style="%s: %s">%s</span>' %(mode_str, COLORS[clss], word) + ' '                
    
    return doc_str

def get_url(scraped_df, doc):
    url = scraped_df[scraped_df['text'] == doc].url.to_list()[0]
    return url, get_url_title(url)


def get_url_title(url):
    first, second = '', ''
    sp = url.split('/')
    for s in sp:
        if 'www' in s:
            s = url.split('.')[1]
            s = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', ' ', s)
            first = s.title()
            
    if first == '':
        s = sp[2]
        s = url.split('.')[1]
        s = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', ' ', s)
        first = s.title()
        
    second = sp[-2].title()

    return '%s | %s' %(first, second)


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


def get_query_word_relations(encoding, query, doc, query_lang='en', doc_lang='en'):
    split_imp = []
    sp = query.split(' ')
    for query_new in sp:
        query_vector = encoding.encode(sent=[query_new], lang=query_lang)

        if not np.any(query_vector):
            continue

        tokens_import = get_token_import_doc(encoding=encoding, 
                                                     query_vec=query_vector, 
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