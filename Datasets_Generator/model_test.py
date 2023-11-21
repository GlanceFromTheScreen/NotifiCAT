import spacy
from model_prediction import classify_entities
from data_parsing import get_dict_of_data

##############################
# TESTING MODEL EXAMPLE #
##############################

nlp = spacy.load("en_core_web_sm")
nlp_ner = spacy.load("model-best")
sentence = "remind me to catch the bus tomorrow at 13:45"

a1 = classify_entities(sentence, nlp_ner)
a2 = classify_entities("remind me to see the cartoon the day after tomorrow at 5:45", nlp_ner)
a3 = classify_entities("remind me to feed cat in 30 minutes", nlp_ner, nlp)
print(a1)
print(a2)
print(a3)

print()

print(get_dict_of_data(a1))
print(get_dict_of_data(a2))
print(get_dict_of_data(a3))
