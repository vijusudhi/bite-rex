import torch
from umap import UMAP
import pandas as pd
import plotly.express as px
import textwrap

from tqdm.notebook import tqdm as tq

def generate_tsv_files(w2v):
    vec_file = open('vectors_%s.tsv'%(w2v.name), 'w')
    words_file = open('words_%s.tsv'%(w2v.name), 'w')
    for word in tq(w2v.wv_vocab):
        # vec = w2v.model.wv[word]
        vec = w2v.model.syn1neg[w2v.wv_vocab.index(word)] + w2v.model.wv[word]
        vec_str = ''
        for i in vec:
            vec_str += str(i) + '\t'
        vec_file.write(vec_str.strip()+ '\n')
        words_file.write(word.strip()+ '\n')
    vec_file.close()
    words_file.close()
    print('Vectors saved at: %s' %('vectors_%s.tsv'%(w2v.name)))
    print('Words saved at: %s' %('words_%s.tsv'%(w2v.name)))
    
    
def generate_emb_files(w2v):
    vec_file = open('vectors_%s.emb'%(w2v.name), 'w')
    words_file = open('words_%s.emb'%(w2v.name), 'w')
    for word in tq(w2v.wv_vocab):
        # vec = w2v.model.wv[word]
        vec = w2v.model.syn1neg[w2v.wv_vocab.index(word)] + w2v.model.wv[word]
        vec_str = ''
        for i in vec:
            vec_str += str(i) + '\t'
        vec_file.write(vec_str.strip()+ '\n')
        words_file.write(word.strip()+ '\n')
    vec_file.close()
    words_file.close()
    print('Vectors saved at: %s' %('vectors_%s.tsv'%(w2v.name)))
    print('Words saved at: %s' %('words_%s.tsv'%(w2v.name)))
    
    
def get_projection_2d(vectors):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(vectors)
    return proj_2d


def plot_umap_2d(docs, proj_2d, colors=None):
    df = pd.DataFrame(docs)
    df.columns = ['text']
    
    fig_2d = px.scatter(
                        proj_2d, 
                        x=0, 
                        y=1,
                        hover_name=df['text'].apply(
                            lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))
                            )
                        )

    fig_2d.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
    
    if not colors:
        fig_2d.update_traces(hoverlabel=dict(align="left"))
    else:
        fig_2d.update_traces(hoverlabel=dict(align="left"), marker_color=colors)
    
    fig_2d.show()    