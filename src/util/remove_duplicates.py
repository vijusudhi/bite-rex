from difflib import SequenceMatcher

def remove_duplicates(docs):
    duplicates = []
    for idx, doc_1 in enumerate(docs):
        for idx_2, doc_2 in enumerate(docs):
            if idx != idx_2:
                if SequenceMatcher(a=doc_1,b=doc_2).ratio() > 0.7:
                    duplicates.append((idx, idx_2))
    duplicate_idx = [s[1] for s in list(set([tuple(sorted(sub)) for sub in duplicates]))]
    print(duplicate_idx)
    
    return duplicate_idx

def filter(docs, sims):
    duplicate_idx = remove_duplicates(docs)
    docs_f, sims_f = [], []
    for idx, (doc, sim) in enumerate(zip(docs, sims)):
        if idx in duplicate_idx:
            continue
        docs_f.append(doc)
        sims_f.append(sim)
    return docs_f, sims_f

def remove_and_filter(retrieved, num_retrieval):
    en_docs, en_sim = filter(retrieved['en_docs'], retrieved['en_sim'])
    de_docs, de_sim = filter(retrieved['de_docs'], retrieved['de_sim'])   
    
    filtered = {
        'en_docs': en_docs[:num_retrieval],
        'en_sim': en_sim[:num_retrieval],
        'de_docs': de_docs[:num_retrieval],
        'de_sim': de_sim[:num_retrieval],
    }

    return filtered