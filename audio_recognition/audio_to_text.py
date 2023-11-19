import whisper

if __name__ == '__main__':
    model = whisper.load_model('base')
    res = model.transcribe('feed.m4a', fp16=False)

    print(res['text'])
