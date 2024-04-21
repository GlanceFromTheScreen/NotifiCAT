import pickle

text = "Iraq's military reported no drones or fighter jets in the area before or during the blast.But the militia organisation involved, the Popular Mobilisation Forces (PMF), blamed an attack.It comes amid heightened tensions between Israel and Iran.The PMF is an umbrella organisation that contains several pro-Iranian militias that are part of the network of proxies that Tehran has been able to use to advance its interests across the Middle East for years.Those militias have attacked US personnel and interests in Iraq on a number of occasions in the past, with the aim of trying to force the remaining US troops to leave the country. The US has regularly struck back at the militias.The Pentagon has, however, moved swiftly to deny any involvement in this incident.It comes as the long-running confrontation between Israel and Iran has recently moved into a new, more perilous phase, amid tensions heightened by the war in Gaza.In contrast to the shadow war they have been engaged in for many years, both countries appear to have directly attacked each other's territory.There was a presumed Israeli strike carried out near the Iranian city of Isfahan early on Friday morning.The attack was an apparent response to a major drone and missile barrage that Tehran launched on Israel six days earlier.The Iranian action was in retaliation for an earlier strike on senior Iranian military commanders in the Syrian capital Damascus - widely believed to have been carried out by Israel."

import torch

input_file = r'summarization/model_clean_save_torch'
summarization_clean_torch_model = torch.load(input_file, map_location=torch.device('cpu'))

print(summarization_clean_torch_model.predict([text])[0])
