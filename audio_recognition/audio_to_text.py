import whisper

if __name__ == '__main__':
    model = whisper.load_model('base')
    res = model.transcribe('123.wav', fp16=False)

    print(res['text'])
