import re
import gensim
from fse.models import Average
from fse import IndexedList

from util import util

def get_doc_train(doc):
    doc = doc.lower()
    doc = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', '', doc)
    doc = doc.split(' ')
    return doc

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


def encode(keyed_vec, docs, path):
    # keyed vectors from gensim is the input
    model = Average(keyed_vec)
    
    doc_list, doc_tuple = get_doc_list_and_tuple(docs)
    
    # create an index
    indexed = IndexedList(doc_list)
    
    # train the index
    model.train(indexed)   
    
    # get document vectors
    doc_vectors = get_doc_vectors(model, doc_tuple)
    
    # save the vectors
    util.compress_pickle(path, doc_vectors)
    # util.compress_pickle(path+'_model', model)