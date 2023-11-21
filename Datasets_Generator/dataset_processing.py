from new_dataset import dataset
import pandas as pd
from spacy.tokens import DocBin
from tqdm import tqdm
import spacy
from spacy.util import filter_spans

##############################
# EXECUTE THIS FILE TO CONSTRUCT TRAINING DATA #
##############################

dataset = pd.DataFrame(dataset)
col_names = ['text', 'date', 'time', 'idea', 'rubbish']
dataset.columns = col_names

##############################
# VARIANT 1 - NO TIME AND DATE IN ENTITIES TUPLES #
##############################

training_data = []
for row in dataset.itertuples():
    text = row.text[0]
    dict_ = {"text": text}
    ent_list = []
    for item in row.idea:
        ent_list.append((text.find(item), text.find(item) + len(item), 'NTFY'))

    dict_['entities'] = ent_list
    training_data.append(dict_)

##############################
# VARIANT 2 - WITH TIME AND DATE IN ENTITIES TUPLES #
##############################

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

##############################
# END OF VARIANTS #
##############################

##############################
# NOW WE NEED MAKE A BINARY FILE FROM DATASET #
##############################

nlp = spacy.load("en_core_web_sm")  # use pre-trained model (you have to load in advance)
doc_bin = DocBin()

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

doc_bin.to_disk("train.spacy")




