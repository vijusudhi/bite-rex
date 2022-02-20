import re

file = open("/home/sudhi/thesis/thesis_cltr_app/apps/util/split_compounds.txt", "r")
lines = file.readlines()
file.close()

comp_word = {}
for line in lines[2:]:
    sp = line.split("\t")
    comp_word[sp[0].strip().lower()] = [s.lower() for s in sp[1:]]

"""
abridged from https://github.com/repodiac/german_compound_splitter/blob/master/german_compound_splitter/comp_split.py
"""
# selection of common prefixes
MERGE_LEFT = ['ab', 'an', 'auf', 'aus', 'außer', 'be', 'bei', 'binnen', 'dar', 'dran', 'durch', 'ein', 'ent', 'er',
              'fehl', 'fort', 'frei', 'ge', 'her', 'hin', 'hinter', 'hoch', 'miss', 'mit', 'nach', 'ober',
              'tief', 'über', 'um', 'un', 'unter', 'ur', 'ver', 'voll', 'vor', 'weg', 'zer', 'zu', 'zur']

# selection of common suffixes
MERGE_RIGHT = ['heit', 'keit', 'schaft', 'tion',  'euse', 'chen', 'lein',
               'ung', 'ion', 'eur', 'ent', 'ant', 'ist', 'oge', 'ine', 'nis', 'ium', 'mus'
               'ff', 'au', 'ei', 'um',  
               'er', 'el', 'or', 
               'us', 'e',  'ur', # 't',
               'ar', 'a', 'ie'  # 'in', 'ät'
            ]

PLURAL_SUFFIX = ['en', 'er', 'e', 'n', 's']


def split_compound(word):
    if word.lower() in comp_word.keys():
        return [w.strip() for w in comp_word[word]]
    else:
        return [word]

def remove_accents(word):
    for accent in MERGE_RIGHT:
        if re.findall(f'{accent}$', word):
            word = re.sub(f'{accent}$', '', word)
            break
    # for accent in PLURAL_SUFFIX:
    #     word = re.sub(f'{accent}$', '', word)     
    for accent in MERGE_LEFT:
        if re.findall(f'^{accent}', word):
            word = re.sub(f'^{accent}', '', word)
            break
    return word

