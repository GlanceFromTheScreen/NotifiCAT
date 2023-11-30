import whisper
from audio_recognition.audio_to_text import audio_recognition
import spacy
from model_prediction import classify_entities
from data_parsing import get_dict_of_data
from audio_to_notification_pipeline import pipeline


##############################
# TESTING MODELS EXAMPLES #
##############################


nlp = spacy.load("en_core_web_sm")
nlp_ner = spacy.load("model-best")

whisper_model = whisper.load_model('base')

sentence = "remind me to catch the bus tomorrow at 13:45"
sentence_whisper = audio_recognition('../audio_recognition/feed.m4a', whisper_model)

a1 = classify_entities(sentence, nlp_ner)
a2 = classify_entities("remind me to see the cartoon the day after tomorrow at 5:45", nlp_ner)
a3 = classify_entities("remind me to feed cat in 30 minutes", nlp_ner, nlp)
a4 = classify_entities(sentence_whisper, nlp_ner, nlp)
a5 = classify_entities("remind me to feed cat at half past three", nlp_ner, nlp)
a6 = classify_entities("remind me tomorrow at 10.30 to go walking with my friends.", nlp_ner, nlp)

print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print(a6)

print()

print(get_dict_of_data(a1))
print(get_dict_of_data(a2))
print(get_dict_of_data(a3))
print(get_dict_of_data(a4))
print(get_dict_of_data(a5))

print()

print(pipeline('../audio_recognition/feed.m4a', whisper_model, nlp_ner, nlp))
