import gensim
import os
import numpy as np
from tempfile import TemporaryFile
outfile = TemporaryFile()

with open('/media/rishabh/dump_bin/Animals_with_Attributes2/Test/list.txt') as f:
    awa_testclass = f.readlines()

print(awa_testclass)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/rishabh/mnist/GoogleNews-vectors-negative300.bin.gz', binary=True)
for i in awa_testclass:
    x=i[:-1]
    features=model[x]
    print(features)
    np_name = x
    np_name = np_name + ".npy"
    np.save(os.path.join('/media/rishabh/dump_bin/Animals_with_Attributes2/Test/',np_name),features)

