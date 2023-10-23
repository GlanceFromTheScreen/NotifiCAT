import nemo
import nemo.collections.asr as nemo_asr

#импорт предобученной модели на русском языке, которая дана в репозитории nemo

if __name__ == '__main__':
    rus_quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained (model_name='stt_ru_quartznet15x5')
