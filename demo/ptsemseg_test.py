from ptsemseg.models import get_model

model_dict = {'arch':'unet'}
model = get_model(model_dict,3)
print(model)

