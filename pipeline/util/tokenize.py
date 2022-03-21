import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.cistem import Cistem
import re

# define lemmatizer for EN and DE
lemmatizer_en = WordNetLemmatizer()
lemmatizer_de = Cistem()

# define stop words for EN and DE
stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))

# define punctuations
punct = list(string.punctuation)

def get_tokens(text, lang='en', lemmatize=False):
    '''
    This function tokenizes and returns the unique tokens
    '''
    text = text.lower()
    text = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', ' ', text)   
    tokens = nltk.word_tokenize(text)
    
    stop_words = stop_words_en if lang=='en' else stop_words_de
    lemmatizer = lemmatizer_en.lemmatize if lang=='en' else lemmatizer_de.stem
    
    pre_proc_tokens = []
    for token in tokens:
        token = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', ' ', token)        
        if token == ' ' or token.isdigit() or token in stop_words or token in punct:
            continue
        
        if lemmatize:
            lemma = lemmatizer(token)
            pre_proc_tokens.append(lemma)
        else:
            pre_proc_tokens.append(token)
    
    pre_proc_tokens = list(set(pre_proc_tokens))
    return pre_proc_tokens


def get_naive_tokens(text):
    doc = text.lower()
    doc = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', '', doc)
    doc = re.sub(r'\.', '', doc)
    tokens = doc.split(' ')
    return tokens