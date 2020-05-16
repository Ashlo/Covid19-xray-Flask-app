from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('model_vgg16.h5')
img  = image.load_img('dataset/Train/covid/1-s2.0-S0929664620300449-gr2_lrg-c (11th copy).jpg',target_size = (224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
result  = classes.astype(int)
result = result.tolist()

print(result)
if result[0][0] == 0:
    print("Negative")
elif result[0][0] == 1:
    print("positive")
