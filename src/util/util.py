import bz2
import _pickle as cPickle

# Pickle a file and then compress it into a file with extension 
def compress_pickle(name, data):
    with bz2.BZ2File("%s.pbz2"%name, 'w') as f: 
        cPickle.dump(data, f)
        

# Load compressed pickle file
def decompress_pickle(name):
    data = bz2.BZ2File("%s.pbz2"%name, 'rb')
    data = cPickle.load(data)       
    return data