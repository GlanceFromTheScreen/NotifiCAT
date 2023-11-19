import json
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

VECTOR_SIZE = 25
json_file_path = 'dataset_1_2.json'

'''initialize word2vec model 1'''

# model = Word2Vec(sentences=common_texts, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)
# model2 = gensim.downloader.load('glove-wiki-gigaword-50')
# model2.save("fstwk.d2v")

model25 = gensim.models.KeyedVectors.load("fstwk.d2v")

print(model25)


'''construct a dataset for classification on data / time / ...'''

df = pd.read_json(json_file_path)
df['sentence'] = df['sentence'].apply(lambda x: x.lower())
df['split'] = df['sentence'].apply(lambda x: x.split())

my_model = Word2Vec(sentences=df['split'],
                    min_count=1,
                    vector_size=50,
                    window=5)

Test = df.drop('sentence', axis=1).drop('split', axis=1)
Test_melted = pd.melt(Test, value_vars=Test.columns, var_name='Target', value_name='Factor')


def count_phrase(x):
    if len(x) == 0:
        return np.zeros(VECTOR_SIZE)
    l = x[0].lower().split()
    buffer = np.zeros(VECTOR_SIZE)
    for item in l:
        buffer = np.array(buffer) + np.array(model25.wv[item])
    return buffer


Test_melted['Super_Factor'] = Test_melted['Factor'].apply(lambda x: count_phrase(x))
Test_melted_exp = pd.DataFrame(Test_melted['Super_Factor'].apply(pd.Series))


'''classification'''

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Test_melted_exp, Test_melted['Target'])


print(neigh.predict_proba(model25.wv['remind'].reshape(1, -1)))






