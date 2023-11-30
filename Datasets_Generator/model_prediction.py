##############################
# MODELS VARIANTS
# nlp = spacy.load("en_core_web_sm")
# nlp_ner = spacy.load("model-best")
# do not forget to import spacy
#
# SENTENCE EXAMPLE
# sentence = "remind me to catch the bus tomorrow at 13:45"
##############################


def clean_sentence(sentence):
    sentence = sentence[:-1]
    return sentence.replace(".", ":")


def classify_entities(sentence, main_model, aux_model=None):
    sentence = clean_sentence(sentence)
    entities = []
    doc = main_model(sentence)
    if aux_model is None:  # use only my model
        for ent in doc.ents:
            entities.append([ent.text, ent.label_])
    else:
        doc_aux = aux_model(sentence)
        for ent in doc.ents:
            if ent.label_ == 'NTFY':
                entities.append([ent.text, ent.label_])
        for ent in doc_aux.ents:
            entities.append([ent.text, ent.label_])

    return entities



