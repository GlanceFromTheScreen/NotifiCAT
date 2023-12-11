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
nlp_ner = spacy.load("model-last")

whisper_model = whisper.load_model('base')

sentence = "remind me to catch the bus tomorrow at 13:45"
sentence_whisper = audio_recognition('../audio_recognition/feed.m4a', whisper_model)

a1 = classify_entities(sentence, nlp_ner)
a2 = classify_entities("remind me to see the cartoon the day after tomorrow at 5:45", nlp_ner)
a3 = classify_entities("remind me to feed cat in 30 minutes", nlp_ner, nlp)
a4 = classify_entities(sentence_whisper, nlp_ner, nlp)
a5 = classify_entities("I would like to go to Moscow tomorrow at 12:12", nlp_ner, nlp)
a6 = classify_entities("plan to visit Moscow tomorrow at 7:30.", nlp_ner, nlp)

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
print(get_dict_of_data(a6))


print()

print(pipeline('../audio_recognition/feed.m4a', whisper_model, nlp_ner, nlp))
