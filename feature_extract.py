from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import pylab

base_model = VGG19(weights='imagenet',include_top=False, pooling='max')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('').output)

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = base_model.predict(x)
print(features.shape)

print(features.tolist())
