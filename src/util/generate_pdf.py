from fpdf import FPDF
from datetime import datetime
import sys
sys.path.append('../')
from util import tokenize, explain
from wordcloud import WordCloud

def get_pdf(query, scraped_df, retrieved):
    pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
    pdf.add_page()

    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Header
    pdf.image(name='data/icon.png', x=10, y=0, w = 30, h = 30)
    pdf.set_font('arial', style='b', size=14)
    pdf.text(50, 10, 'Explainable Cross-Lingual Text Retrieval on Automotive Domain')
    pdf.set_font('arial', style='', size=10)
    pdf.text(50, 15, f'Report generated at: {time_str}')
    pdf.line(50, 20, 200, 20)

    # Search query
    pdf.set_font('arial', style='b', size=12)
    pdf.text(70, 30, f'Search query: \'{query}\'')

    # Wordclouds
    pdf.image(name='wordcloud_en.png', x=50, y=40, w=50, h=40)
    pdf.image(name='wordcloud_de.png', x=110, y=40, w=50, h=40)
    
    # group docs as per titles of companies
    titles = []
    docs = []
    langs = []
    urls = []
    for doc in retrieved['en_docs']:
        url, title = explain.get_url(scraped_df, doc)
        titles.append(title.split('|')[0].strip())
        docs.append(doc)
        langs.append('en')
        urls.append(url)
    for doc in retrieved['de_docs']:
        url, title = explain.get_url(scraped_df, doc)
        titles.append(title.split('|')[0].strip())
        docs.append(doc)
        langs.append('de')
        urls.append(url)

    set_titles = list(set(titles))

    docs_titles = [''] * len(set_titles)
    for ind, title in enumerate(set_titles):   
        for d, t in zip(docs, titles):
            if title == t:
                docs_titles[ind] += d

                
    # plot wordclouds of each company
    y_start = 100
    for ind, title in enumerate(set_titles):
        title = title.encode('latin-1', 'ignore').decode('latin-1')
        pdf.set_font('arial', style='b', size=8)
        
        tokens = ''
        for idx, t in enumerate(titles):
            if title == t:
                tokens += ' '.join(tokenize.get_tokens(text=docs[idx], lang=langs[idx]))

        wordcloud = WordCloud(background_color='white').generate(tokens)
        wordcloud.to_file('wordcloud_%s.png'%title)       

        if ind % 2 == 0:
            pdf.image(name='wordcloud_%s.png'%title, x=50, y=y_start, w=50, h=40)
            pdf.set_xy(50, y_start-5)
            pdf.multi_cell(w=50, h=5, txt=title, border=0, 
                                align='C', fill=False)

        else:
            pdf.image(name='wordcloud_%s.png'%title, x=110, y=y_start, w=50, h=40)
            pdf.set_xy(110, y_start-5)
            pdf.multi_cell(w=50, h=5, txt=title, border=0, 
                                align='C', fill=False)

            y_start += 50                    

    # display docs of each company
    pdf.add_page()
    y_start = 10
    for ind, title in enumerate(set_titles): 
        title = title.encode('latin-1', 'ignore').decode('latin-1')
        pdf.set_font('arial', style='b', size=8)    
        pdf.set_xy(20, y_start)
        pdf.multi_cell(w=50, h=5, txt=title, border=0, 
                        align='L', fill=False)   

        for ind, t in enumerate(titles):
            if title == t:
                doc = docs[ind].encode('latin-1', 'ignore').decode('latin-1')
                pdf.set_font('arial', style='', size=8)   
                pdf.set_xy(70, y_start)
                pdf.multi_cell(w=110, h=5, txt=f'{doc}', border=0, 
                                align='J', fill=False)
                y = pdf.get_y()
                y_start = y+5


    pdf.output(f'report_{query}.pdf')            