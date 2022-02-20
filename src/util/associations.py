from util import display, significance


def get_query_word_relations(model, query, doc, query_lang='en', doc_lang='en'):
    split_imp = []
    sp = query.split(' ')
    for query_new in sp:
        tokens_import = significance.get_token_import_doc(model=model, 
                                                     query=query_new, 
                                                     doc=doc,
                                                     lang=doc_lang
                                                    )

        txt = display.get_display_text(doc, tokens_import, mode='background')
        
        
        split_imp.append(
            {
                'split': query_new,
                'text': txt
            }
        )
        
    return split_imp