from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from util import util, tokenize


class CorpusCountVectorizer():
    def __init__(self, path):
        epo_df = util.decompress_pickle(path)
        corpus = []
        for en, de in zip(epo_df.en, epo_df.de):
            corpus.append(en + " " +  de)
        vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=100000)
        self.X = vectorizer.fit_transform(corpus).toarray()
        self.vocab = vectorizer.get_feature_names()

    def get_index(self, word):
        ind = -1
        try:
            ind = self.vocab.index(word)
        except:
            pass    
        return ind


    def get_occurence(self, ind_1, ind_2=None):
        # TODO: instead of iterating through each x,
        # convert to numpy arrays, mask and count non zeros.
        # this could fasten the function
        count = 0
        for x in self.X:
            if ind_2:
                if x[ind_1]!=0 and x[ind_2]!=0:
                    count += 1
            else:
                if x[ind_1]!=0:
                    count += 1
        return count


    def get_cooccur_matrix(self, query, query_lang, doc, doc_lang, plot=False):
        query_tokens = tokenize.get_tokens(query, lang=query_lang)
        doc_tokens = tokenize.get_tokens(doc, lang=doc_lang)

        matrix = []
        for token in query_tokens:
            ind = self.get_index(token)
            primary_count = self.get_occurence(ind_1=ind)
            row = []
            for word in doc_tokens:
                ind_2 = self.get_index(word)
                count = self.get_occurence(ind_1=ind, ind_2=ind_2)
                co_occurence = (count / primary_count) * 100
                row.append(co_occurence)
            matrix.append(row)

        if plot:
            return self.plot_heatmap(matrix, query_tokens, doc_tokens)
        else:
            return matrix

    @staticmethod
    def plot_heatmap(matrix, query_tokens, doc_tokens):
        fig = px.imshow(matrix, 
                        labels=dict(x="Document term", y="Query term", color="Co-occurence %"),
                        y=query_tokens,
                        x=doc_tokens,
                        text_auto=".2f", aspect="auto",
                        color_continuous_scale=px.colors.sequential.Reds
                        )
        return fig