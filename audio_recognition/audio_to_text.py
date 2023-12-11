##############################
# MODELS VARIANTS
# model = whisper.load_model('base')
# do not forget to import whisper
#
# SENTENCE EXAMPLE
# sentence = "remind me to catch the bus tomorrow at 13:45"
##############################

def audio_recognition(file_path, model):
    res = model.transcribe(file_path, fp16=False, language='en')
    return res['text']




