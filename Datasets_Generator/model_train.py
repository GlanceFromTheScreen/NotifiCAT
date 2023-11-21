##############################
# 1) create config (RUN NEXT COMMAND IN TERMINAL)
#     python -m spacy init fill-config base_config.cfg config.cfg
# 2) configure config.cfg file. Recommended to adjust:
#     - batch_size = 100
#     - max_epochs = 20
# 3) train model (it uses train.spacy that we got after processing dataset) (RUN NEXT COMMAND IN TERMINAL)
#     python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./train.spacy
##############################

