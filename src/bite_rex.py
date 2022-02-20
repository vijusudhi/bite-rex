import streamlit as st
import datetime
import time
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from sentence_transformers import SentenceTransformer
import torch 
import re
import os
import numpy as np

import sys
from util import util, tokenize, vectorize, explain_cont, states, query_suggestions, corpus_count_vectorizer, explain_repr_space, remove_duplicates

import random

NUM_RETRIEVAL = 5

st.set_page_config(page_title='BiTe-REx: Retrieve and Explain!', page_icon = "🦖", layout = 'centered', initial_sidebar_state = 'collapsed')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("data/css/style.css") 

DE_DOCS = '1BHIR_AmbyII1wZ5wWbobR4nQU0AYoAAN'
EN_DOCS = '1hSF-CzOydd0zfO7BkMfT8UoT486nabQO'

DE_WV_UNSUP = '1T6R0NwZT4EGafu7WadSOR6ceCfbumgEQ'
EN_WV_UNSUP = ''

DE_VECTORS = ''
EN_VECTORS = ''

CLS_MODEL = ''
TFIDF_MODEL = ''
CONT_MODEL_MLM_NSP = '1lpl3V-iaaKmwR7myHsuqPMX82Tdq_HHU'


@st.cache(allow_output_mutation=True, ttl=3600)
def load_model():
    def download_file(file_path, drive_link, unzip=False):
        file = Path(file_path)
        if not file.exists():
            with st.spinner("Downloading %s. Please wait..." % file_path):
                gdd.download_file_from_google_drive(file_id=drive_link,
                                        dest_path=file_path,
                                        unzip=unzip)             
    
    download_file(file_path='data/docs/scraped_de_docs.pbz2', drive_link=DE_DOCS, unzip=False)  
    download_file(file_path='data/docs/scraped_en_docs.pbz2', drive_link=EN_DOCS, unzip=False)  
    download_file(file_path='data/static/de_wv_unsup.pbz2', drive_link=DE_WV_UNSUP, unzip=False)    
    download_file(file_path='data/static/en_wv_unsup.pbz2', drive_link=EN_WV_UNSUP, unzip=False)   
    download_file(file_path='data/static/st_de_doc_vectors.pbz2', drive_link=DE_VECTORS, unzip=False)    
    download_file(file_path='data/static/st_en_doc_vectors.pbz2', drive_link=EN_VECTORS, unzip=False)    
    download_file(file_path='data/models/model_MLM_NSP.pt', drive_link=CONT_MODEL_MLM_NSP, unzip=False)
    download_file(file_path='data/models/cls_model.pbz2', drive_link=CLS_MODEL, unzip=False)
    download_file(file_path='data/models/tfidf_vc.pbz2', drive_link=TFIDF_MODEL, unzip=False)

@st.cache(allow_output_mutation=True, ttl=3600)
def load_states():
    scraped_df = util.decompress_pickle('data/scraped_df/scraped_df')
    docs = scraped_df.text.to_list()

    cls_model = util.decompress_pickle('data/models/cls_model')
    tfidf_vc = util.decompress_pickle('data/models/tfidf_vc')

    scraped_en_docs = util.decompress_pickle('data/docs/scraped_en_docs') 
    scraped_de_docs = util.decompress_pickle('data/docs/scraped_de_docs')          
    
    # static model
    st_en_doc_vectors = util.decompress_pickle('data/static/st_en_doc_vectors')
    st_de_doc_vectors = util.decompress_pickle('data/static/st_de_doc_vectors')
    
    scraped_en_docs_, st_en_doc_vectors_ = [], []
    for doc, vec in zip(scraped_en_docs, st_en_doc_vectors):
        if len(doc.split(' ')) < 10:
            continue
        scraped_en_docs_.append(doc)
        st_en_doc_vectors_.append(vec)
            
    scraped_de_docs_, st_de_doc_vectors_ = [], []
    for doc, vec in zip(scraped_de_docs, st_de_doc_vectors):
        if len(doc.split(' ')) < 10:
            continue
        scraped_de_docs_.append(doc)
        st_de_doc_vectors_.append(vec)    

    en_wv_unsup = util.decompress_pickle('data/static/en_wv_unsup')
    de_wv_unsup = util.decompress_pickle('data/static/de_wv_unsup')
    
    st_encoding = vectorize.Encoding(model=[en_wv_unsup, de_wv_unsup], model_type='static')
    st_retrieval = vectorize.Retrieval(vectors=[st_en_doc_vectors_, st_de_doc_vectors_],
                                    docs=[scraped_en_docs_, scraped_de_docs_]
                                   )
    
    # contextual model
    cont_model = torch.load('data/models/model_MLM_NSP.pt').module.eval()
    print('loaded', len(scraped_en_docs), len(scraped_de_docs))


    # count vectorizer 
    count_vectorizer = corpus_count_vectorizer.CorpusCountVectorizer(path='/home/sudhi/thesis/master_thesis_cltr/data/other/epo_10_20_df_wo_pre')
    
    cached_state = states.CachedState(
                                scraped_df = scraped_df,
                                scraped_docs = [scraped_en_docs_, scraped_de_docs_],
                                st_encoding = st_encoding,
                                st_retrieval = st_retrieval,        
                                cont_model = cont_model,
                                cls_model = cls_model,
                                tfidf_vc = tfidf_vc,
                                count_vectorizer = count_vectorizer
                            )
    
    return cached_state


def display_header():
    col1, col2 = st.columns([20,20])
    col1.markdown("<span class='title big'>🦖 BiTe-REx</span>", unsafe_allow_html=True)
    col2.markdown("<span class='title small'>**Bi**lingual **Te**xt **R**etrieval **Ex**planations<br>in the Automotive domain</span>", unsafe_allow_html=True)
    
    if 'model_idx' in st.session_state:
        model_idx = st.session_state.model_idx
        num_retrieval = st.session_state.num_retrieval
    else:
        model_idx = 0
        num_retrieval = 5
        
    # Sidebar
    model_type = st.sidebar.selectbox(
        label='Embedding model',
        options=('Static Embeddings', 'Contextual Embeddings'),
        index=model_idx
    )

    num_retrieval = st.sidebar.slider(
        label='Select number of documents to be retrieved',
        min_value=3, 
        max_value=10, 
        value=num_retrieval
    )
    
    model_idx = 0 if model_type=='Static Embeddings'else 1
    
    st.session_state.update(
                    {
                     'model_idx':model_idx,
                     'model_type':model_type,
                     'num_retrieval':num_retrieval
                    }
    )


def page_home():
    if 'app_state' in st.session_state:
        app_state = st.session_state['app_state']
        
        query_text = app_state.query
        model_type = st.session_state.model_type
        model_idx = st.session_state.model_idx
        num_retrieval = st.session_state.num_retrieval
        model = app_state.model
        query = app_state.query
        query_lang = app_state.query_lang
        query_lang_corrected = app_state.query_lang_corrected
        retrieved = app_state.retrieved
        token_importance_en = app_state.token_importance_en
        token_importance_de = app_state.token_importance_de
    else:
        model_idx = 0
        num_retrieval = 5
        query_text = ''
        query_lang_corrected = False
        
        model_type = st.session_state.model_type
        model_idx = st.session_state.model_idx
        num_retrieval = st.session_state.num_retrieval

    with st.spinner('Please wait while we load the environment..'):
        load_model()
        cached_state = load_states()
    
    # if model_type == 'Static Embeddings':
    #     encoding = cached_state.st_encoding
    #     retrieval = cached_state.st_retrieval
    # else:
    # encoding = cached_state.cont_encoding
        
    model = cached_state.cont_model
    docs = cached_state.scraped_docs
        
    st.write('### Enter the query')       
    
    query = st.text_input(label='Enter the query and type Ctrl+Enter to search',
                         value=query_text)
    
    if query != '':
        if query_text != query:
            query_lang_corrected = False

        query_us = re.sub(' ', '_', query) 

        test = cached_state.tfidf_vc.transform([query.lower()])
        pred = cached_state.cls_model.predict(test)
        query_lang = 'en' if pred == 0 else 'de'

        if not os.path.exists(f"dump/{query_us}"):
            os.makedirs(f"dump/{query_us}")

        util.compress_pickle(f"dump/{query_us}/query_lang", query_lang)

        with st.container():
            col1, col2 = st.columns([5, 5])
            with col1:
                st.markdown(f"Query language identified as: <span class='highlight red_bold'>{query_lang}</span>", unsafe_allow_html=True)    
            with col2:
                lang_change_button = st.button(label="Change", key='btn_change_lang')
                if lang_change_button or query_lang_corrected:
                    query_lang = 'en' if query_lang == 'de' else 'de'
                    query_lang_corrected = True
                    with col1:
                        st.markdown(f"Sorry! Query language corrected to: <span class='highlight red_bold'>{query_lang}</span>", unsafe_allow_html=True) 

        now = datetime.datetime.now()   

        suggest = query_suggestions.SuggestQueries(model=model)
        suggestions = suggest.get_suggestions(query)
        suggestions.sort(key=lambda s: len(s))
        util.compress_pickle(f"dump/{query_us}/suggestions", suggestions)
        st.write("### You may also search for")
        with st.container():
            text = ""
            length = 0
            for suggestion in suggestions:
                length += len(suggestion)
                # check if the length of string exceeds container size
                # add just the span if it does not
                # else, write the text and reinitialize the values
                if length <= 75:
                    text += f"<span class='highlight red'>{suggestion}</span>"
                else:
                    st.markdown(text, unsafe_allow_html=True)
                    text = ""
                    length = len(suggestion)
                    text += f"<span class='highlight red'>{suggestion}</span>"
            # write any pieces of text remaining
            st.markdown(text, unsafe_allow_html=True)

        # if 'app_state' not in st.session_state:
        query_vector = cached_state.st_encoding.encode(sent=[query], lang=query_lang)
        st_retrieved = cached_state.st_retrieval.retrieve(query_vec=query_vector, 
                                                          num_ret=500)
        docs = [st_retrieved['en_docs'], st_retrieved['de_docs']]
        retrieved = vectorize.retrieve_cont(model, query, docs, num_retrieval*2)
        retrieved = remove_duplicates.remove_and_filter(retrieved, num_retrieval)

        util.compress_pickle(f"dump/{query_us}/retrieved", retrieved)

        after = datetime.datetime.now()
        difference = after - now

        st.write('### Search results for "%s" in %f microseconds'%(query, difference.microseconds))
        docs = retrieved['en_docs'] + retrieved['de_docs']

        token_importance_en = explain_cont.get_token_import(model=model, 
                                     query=query,
                                     docs=retrieved['en_docs'],
                                     lang='en'
                                    )
        # print(token_importance_en)
        token_importance_de = explain_cont.get_token_import(model=model, 
                                 query=query,
                                 docs=retrieved['de_docs'],
                                 lang='de'
                                )

        util.compress_pickle(f"dump/{query_us}/token_importance_en", token_importance_en)         
        util.compress_pickle(f"dump/{query_us}/token_importance_de", token_importance_de)                       

        # print(token_importance_de)
        app_state = states.AppState(
                                model_type = model_type,
                                num_retrieval = num_retrieval,
                                model = model,
                                query = query,
                                query_lang = query_lang,
                                query_lang_corrected = query_lang_corrected,
                                retrieved = retrieved,
                                token_importance_en = token_importance_en,
                                token_importance_de = token_importance_de,
                                count_vectorizer = cached_state.count_vectorizer
                            )

        st.session_state['app_state'] = app_state

        display_search_results(cached_state, doc_lang='en')
        display_search_results(cached_state, doc_lang='de')

    
def display_search_results(cached_state, doc_lang):
    if doc_lang == 'en':
        sim = st.session_state['app_state'].retrieved['en_sim']
        docs = st.session_state['app_state'].retrieved['en_docs']
        token_imp = st.session_state['app_state'].token_importance_en
        key_ind = 0
    else:
        sim = st.session_state['app_state'].retrieved['de_sim']
        docs = st.session_state['app_state'].retrieved['de_docs']
        token_imp = st.session_state['app_state'].token_importance_de
        key_ind = 100
    
    for sim, doc in zip(sim, docs):
        html_string = explain_cont.get_display_text(doc, token_imp, mode='bold')
        url, title = explain_cont.get_url(cached_state.scraped_df, doc)
        with st.container():
            col1, mid = st.columns([2, 20])
            explain_state = states.ExplainState(
                                doc=doc,
                                doc_lang=doc_lang,
                                sim=sim
                            )
            st.session_state['explain_state'] = explain_state                            
                            
            with col1:
                button = st.button(label="X", key='%d'%key_ind, 
                                   on_click=update_and_explain,
                                   args=(explain_state,)
                                  )
                key_ind += 1
            with mid:
                st.markdown('[%s](%s)'%(title, url))    
                st.markdown(html_string, unsafe_allow_html=True)
    
def update_and_explain(explain_state):
    st.session_state["page"] = 'Explanations'
    st.session_state['explain_state'] = explain_state   

def page_explanations():    
    encoding = st.session_state['app_state'].encoding
    model = st.session_state['app_state'].model
    query = st.session_state['app_state'].query
    query_lang = st.session_state['app_state'].query_lang
    query_vector = st.session_state['app_state'].query_vector
    count_vectorizer = st.session_state['app_state'].count_vectorizer
    
    sim = st.session_state['explain_state'].sim
    doc = st.session_state['explain_state'].doc
    doc_lang = st.session_state['explain_state'].doc_lang

    cached_state = st.session_state['explain_state'].cached_state
    
    st.write("### Explaining retrieval of:")
    with st.container():
        col1, col2 = st.columns([5, 20])
        col1.markdown("Query")
        col2.markdown(f"<span class='highlight red_bold'>{query}</span>", unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([5, 20])
        col1.markdown("Document")
        col2.markdown(f"**{doc}**", unsafe_allow_html=True)
    
    query_us = re.sub(' ', '_', query)
    retrieved = util.decompress_pickle(f"dump/{query_us}/retrieved")

    if doc in retrieved['en_docs']:
        doc_idx = retrieved['en_docs'].index(doc)
    if doc in retrieved['de_docs']:
        doc_idx = retrieved['de_docs'].index(doc)
    
    # with st.expander("How similar is the document to the query?"):
    #     st.write('This document is similar to your search query by', 
    #              round(sim, 3), '.')
    #     token_imp = explain_cont.get_token_import_doc(model=model, 
    #                                  query=query,
    #                                  doc=doc,
    #                                  lang=doc_lang
    #                                 )
    #     plt_imp = explain_cont.plot_import_bar(token_imp, use_neg=True)

    with st.expander("Explore representation space"):
        ex_repr_space = explain_repr_space.ExplainReprSpace(
                            query=query, query_lang=query_lang,
                            doc=doc,
                            en_ret_docs=retrieved['en_docs'], de_ret_docs=retrieved['de_docs'],
                            scraped_en_docs=cached_state.scraped_docs[0], scraped_de_docs=cached_state.scraped_docs[1], 
                            model=model, st_encoding=cached_state.st_encoding)

        text = f'\
        You can see the representation space of the queries and documents below.<br>\
        <span style="color: transparent;  text-shadow: 0 0 0 green; ">&#9899;</span> Query\
        <span style="color: transparent;  text-shadow: 0 0 0 red; ">&#9899;</span> Document <b>relevant</b> to the query\
        (<span style="color: transparent;  text-shadow: 0 0 0 brown; ">&#9899;</span>if selected)<br>\
        <span style="color: transparent;  text-shadow: 0 0 0 blue; ">&#9899;</span> Document with word \
        <span class="highlight red">{ex_repr_space.imp_word_full}</span><b>not relevant</b> to the query<br>\
        Size of the markers indicate contextual similarity.\
        '
        st.markdown(text, unsafe_allow_html=True)
        if ex_repr_space.repr_space == -1:
            st.markdown('Sorry! Can not dsiplay the space', unsafe_allow_html=True)
        else:                   
            st.plotly_chart(ex_repr_space.repr_space, use_container_width=True)

    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([10, 1, 30])
        col1.markdown("<span class='heading'><b>Query-Document terms co-occurrences</b></span>", unsafe_allow_html=True)
        col2.markdown("<div class= 'vertical'></div>", unsafe_allow_html=True)
        col3.markdown("<p>The model was trained on patents from the European Patent Office (EPO) belonging to the International Patent Classification (IPC) <i>B60 Vehicles in General</i>.</p>",
                     unsafe_allow_html=True
                    )                   
        col3.markdown("<p>You can see the query-document terms co-occurrences found the corpus below.</p>",
                    unsafe_allow_html=True
                    ) 
    heatmap = count_vectorizer.get_cooccur_matrix(query=query, query_lang=query_lang, 
                                                 doc=doc, doc_lang=doc_lang,
                                                 plot=True)
    st.plotly_chart(heatmap, use_container_width=True)        

    # with st.expander("EXP 02 - Query-Document terms co-occurrences"):
    #     st.markdown("<p>The model was trained on patents from the European Patent Office (EPO) belonging to the International Patent Classification (IPC) <i>B60 Vehicles in General</i>.</p> \
    #                  <p>You can see the query-document terms co-occurrences found the corpus below.</p>",
    #                  unsafe_allow_html=True
    #                 )
    #     heatmap = count_vectorizer.get_cooccur_matrix(query=query, query_lang=query_lang, 
    #                                                  doc=doc, doc_lang=doc_lang,
    #                                                  plot=True)
    #     st.plotly_chart(heatmap, use_container_width=True)  
    #   

    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([10, 1, 30])
        col1.markdown("<span class='heading'><b>Query-Document term associations</b></span>", unsafe_allow_html=True)
        col2.markdown("<div class= 'vertical'></div>", unsafe_allow_html=True)
        col3.markdown("<p>The model knows both English and German <i>reasonably well</i>. It can say which pair of words associate with one another.</p>",
                     unsafe_allow_html=True
                    )
        col3.markdown("<p>You can see below <span class='highlight darkbrown_bold'>high</span> to <span class='highlight lightbrown_bold'>low</span> associations of document terms with the query terms.</p>",
                 unsafe_allow_html=True
                )    

    spit_imp = explain_cont.get_query_word_relations(model=model, 
                                                query=query, 
                                                doc=doc, 
                                                query_lang=query_lang, 
                                                doc_lang=doc_lang)  
    st.markdown("<p></p>", unsafe_allow_html=True)   
    with st.container():
        col_q, col_txt = st.columns([5, 20])
        col_q.markdown('*Query term*')
        col_txt.markdown('*Document*')
    for i in spit_imp:
        with st.container():
            # TODO: update column width dynamically according to size
            # of text
            col_q, col_txt = st.columns([5, 20])
            with col_q:
                st.markdown('**%s**' %i['split'])
            with col_txt:
                st.markdown(i['text'], unsafe_allow_html=True)        

    # with st.expander("EXP 03 - Query-Document term associations"):
    #     st.markdown("<p>The model knows both English and German <i>reasonably well</i>. It can say which pair of words associate with one another.</p> \
    #                  <p>You can see below the association of document terms with the query terms, varying from <span class='highlight darkbrown_bold'>high</span> to <span class='highlight lightbrown_bold'>low</span> values.</p>",
    #                  unsafe_allow_html=True
    #                 )        
    #     spit_imp = explain_cont.get_query_word_relations(model=model, 
    #                                                 query=query, 
    #                                                 doc=doc, 
    #                                                 query_lang=query_lang, 
    #                                                 doc_lang=doc_lang)        
    #     with st.container():
    #         col_q, col_txt = st.columns([5, 20])
    #         col_q.markdown('*Query term*')
    #         col_txt.markdown('*Document*')
    #     for i in spit_imp:
    #         with st.container():
    #             # TODO: update column width dynamically according to size
    #             # of text
    #             col_q, col_txt = st.columns([5, 20])
    #             with col_q:
    #                 st.markdown('**%s**' %i['split'])
    #             with col_txt:
    #                 st.markdown(i['text'], unsafe_allow_html=True)   
    #                                    

    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([10, 1, 30])
        col1.markdown("<span class='heading'><b>Document term significance</b></span>", unsafe_allow_html=True)
        col2.markdown("<div class= 'vertical'></div>", unsafe_allow_html=True)
        col3.markdown("<p>Each document term contribute differently to the retrieval of this document. It can either prompt the system to retrieve the document or otherwise.</p>",
                     unsafe_allow_html=True
                    )
        col3.markdown("<p> You can see below the <span class='highlight darkgreen_bold'>positive</span> or <span class='highlight darkred_bold'>negative</span> contribution of document terms to the retrieval.</p>",
                 unsafe_allow_html=True
                )                     

    token_imp = explain_cont.get_token_import_doc(model=model, 
                                 query=query,
                                 doc=doc,
                                 lang=doc_lang
                                )
    plt_imp = explain_cont.plot_import_bar(token_imp, use_neg=True)        
    st.plotly_chart(plt_imp, use_container_width=True)                    

    # with st.expander("EXP 04 - Significance of document terms"):
    #     st.markdown("<p>Each document term contribute differently to the retrieval of this document. It can either prompt the system to retrieve the document or otherwise.</p> \
    #                  <p> You can see below the <span class='highlight darkgreen_bold'>positive</span> or <span class='highlight darkred_bold'>negative</span> contribution of document terms to the retrieval.</p>",
    #                  unsafe_allow_html=True
    #                 )         
    #     token_imp = explain_cont.get_token_import_doc(model=model, 
    #                                  query=query,
    #                                  doc=doc,
    #                                  lang=doc_lang
    #                                 )
    #     plt_imp = explain_cont.plot_import_bar(token_imp, use_neg=True)        
    #     st.plotly_chart(plt_imp, use_container_width=True)

    
    st.session_state["page"] = 'Home'
    exit_button = st.button(label="Exit", key='626', on_click=update_and_exit)

    
def update_and_exit():
    st.session_state["page"] = 'Home'

PAGES = {
    "Home": page_home,
    "Explanations": page_explanations,
}           
    
def main():    
    if "page" not in st.session_state:
        st.session_state.update(
            {
                'page': 'Home'
            }
        )
        
    PAGES[st.session_state['page']]()


DEBUG_MODE = True
if __name__ == "__main__":
    if DEBUG_MODE:
        display_header()
        main()
    else:
        try:
            display_header()
            main()
        except:
            st.error('Oh snap! Error!')