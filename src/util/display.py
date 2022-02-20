from colour import Color
import re
import numpy as np
# import plotly.express as px


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