import spacy
from model_prediction import classify_entities

##############################
# TESTING MODEL EXAMPLE #
##############################

nlp = spacy.load("en_core_web_sm")
nlp_ner = spacy.load("model-best")
sentence = "remind me to catch the bus tomorrow at 13:45"

print(classify_entities(sentence, nlp_ner))
print(classify_entities("remind me to see the cartoon the day after tomorrow at 5:45", nlp_ner))
print(classify_entities("remind me to feed cat in 30 minutes", nlp_ner, nlp))
