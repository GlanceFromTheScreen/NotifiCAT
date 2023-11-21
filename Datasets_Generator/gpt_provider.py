import json
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

VECTOR_SIZE = 25
json_file_path = 'dataset_1_2.json'

'''initialize word2vec model 1'''

# model = Word2Vec(sentences=common_texts, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)
# model2 = gensim.downloader.load('glove-wiki-gigaword-50')
# model2.save("fstwk.d2v")

model25 = gensim.models.KeyedVectors.load("word2vec2.model", mmap='r')
print(model25['hello'])
# print(model25)


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
        try:
            buffer = np.array(buffer) + np.array(model25[item])
        except:
            pass
    return buffer


Test_melted['Super_Factor'] = Test_melted['Factor'].apply(lambda x: count_phrase(x))
Test_melted_exp = pd.DataFrame(Test_melted['Super_Factor'].apply(pd.Series))
print(Test)
print(Test_melted)

scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(Test_melted_exp)
Test_melted_exp_norm = pd.DataFrame(normalized_values, columns=Test_melted_exp.columns)
print(Test_melted_exp_norm)


'''classification'''

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(Test_melted_exp_norm, Test_melted['Target'])


print(neigh.predict(model25['remind'].reshape(1, -1)))


'''work with new dataset'''

# import spacy
#
# # Загрузка модели spaCy для английского языка
# nlp = spacy.load("en_core_web_sm")
#
# # Функция для извлечения дня и времени из текста напоминания
# def extract_day_and_time(text):
#     # Обработка текста с использованием spaCy
#     doc = nlp(text)
#
#     print(doc.ents)
#     for ent in doc.ents:
#         print(ent.text, ent.label_)
#
#     # Извлечение сущностей даты и времени
#     day = None
#     time = None
#
#     for ent in doc.ents:
#         if ent.label_ == "DATE":
#             day = ent.text
#         elif ent.label_ == "TIME":
#             time = ent.text
#
#     return day, time
#
#
# # Пример использования
# reminder_text = "New York is the capital of the country where Jorge Gimmlestone was"
# day, time = extract_day_and_time(reminder_text)
#
# print("Day:", day)
# print("Time:", time)

from new_dataset import dataset

dataset = pd.DataFrame(dataset)
col_names = ['text', 'date', 'time', 'idea', 'rubbish']
dataset.columns = col_names
# dataset = dataset.drop('date', axis=1).drop('time', axis=1).drop('rubbish', axis=1)
print(dataset)

training_data = []
for row in dataset.itertuples():
    text = row.text[0]
    dict_ = {"text": text}
    ent_list = []
    for item in row.idea:
        ent_list.append((text.find(item), text.find(item) + len(item), 'NTFY'))
    for item in row.date:
        ent_list.append((text.find(item), text.find(item) + len(item), 'DATE'))
    for item in row.time:
        ent_list.append((text.find(item), text.find(item) + len(item), 'TIME'))

    dict_['entities'] = ent_list
    training_data.append(dict_)

print(training_data)

from spacy.tokens import DocBin
from tqdm import tqdm
import spacy

# nlp = spacy.blank("en") # load a new spacy model
nlp = spacy.load("en_core_web_sm")
doc_bin = DocBin()

from spacy.util import filter_spans

for training_example in tqdm(training_data):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("train_with_date_time.spacy")

'''now data is processed. now we need to train'''

'''
commands that has to be run in terminal to
1) create config
    python -m spacy init fill-config base_config.cfg config.cfg
2) train model
    python -m spacy train config.cfg --output ./ --paths.train ./train2.spacy --paths.dev ./train2.spacy 
'''

nlp_ner = spacy.load("model-best")
doc = nlp_ner("schedule meeting on December 22 at 15:55")
doc2 = nlp("schedule meeting on December 22 at 15:55")

print()
for ent in doc.ents:
    print(ent.text, ent.label_)






