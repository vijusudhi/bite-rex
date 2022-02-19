import string
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# define stop words for EN and DE
stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))

# define punctuations
punct = list(string.punctuation)

def get_tokens(text, lang='en'):
    '''
    This function tokenizes and returns the unique tokens
    '''
    text = text.lower()
    tokens = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-’]', '', text)
    tokens = re.sub(r'\"', '', text)
    tokens = re.sub(r"\'", '', text)
    tokens = nltk.word_tokenize(tokens)
    
    stop_words = stop_words_en if lang=='en' else stop_words_de
    
    pre_proc_tokens = []
    for token in tokens:        
        if token.isdigit() or token in stop_words or token in punct:
            continue
        
        if token in pre_proc_tokens:
            continue

        pre_proc_tokens.append(token)
            
    return pre_proc_tokens
