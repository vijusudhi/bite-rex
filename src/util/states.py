class CachedState:
    def __init__(self, 
                 scraped_df=None, 
                 cls_model=None, 
                 tfidf_vc=None,                 
                 st_encoding=None, 
                 st_retrieval=None, 
                 cont_encoding=None, 
                 cont_retrieval=None, 
                 cont_model=None,
                 scraped_docs=None,
                 count_vectorizer=None
                ):
        self.scraped_df = scraped_df
        self.scraped_docs = scraped_docs
        self.st_encoding = st_encoding
        self.st_retrieval = st_retrieval
        self.cont_encoding = cont_encoding
        self.cont_retrieval = cont_retrieval
        self.cls_model = cls_model
        self.tfidf_vc = tfidf_vc
        self.cont_model = cont_model
        self.count_vectorizer=count_vectorizer
        
        
class AppState:
    def __init__(self,
                 model_type=None,
                 num_retrieval=None,
                 
                 query=None,
                 query_lang=None,
                 query_lang_corrected=None,
                 
                 retrieved=None,
                 
                 token_importance_en=None,
                 token_importance_de=None,
                
                 query_vector=None,
                 model=None,
                 encoding=None,
                 retrieval=None,   

                 count_vectorizer=None              
                ):
        self.model_type = model_type
        if self.model_type == 'Static Embeddings':
            self.model_idx = 0
        else:
            self.model_idx = 1
        self.num_retrieval = num_retrieval
        self.encoding = encoding
        self.retrieval = retrieval
        self.query = query
        self.query_vector = query_vector
        self.query_lang = query_lang
        self.query_lang_corrected = query_lang_corrected
        self.retrieved = retrieved
        self.token_importance_en = token_importance_en
        self.token_importance_de = token_importance_de
        self.model = model
        self.count_vectorizer = count_vectorizer
        
        
class ExplainState:
    def __init__(self,
                 doc=None,
                 doc_lang=None,
                 sim=None):
        self.doc = doc
        self.doc_lang = doc_lang
        self.sim = sim