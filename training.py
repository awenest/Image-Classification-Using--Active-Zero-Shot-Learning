import os
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np


def start():
    model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', pooling='avg')

    root_dir = "/media/rishabh/dump_bin/Animals_with_Attributes2/JPEGImages/"
    for root, subdirs, files in os.walk(root_dir):
        list_file_path = os.path.join(root, 'list_of_files.txt')
        with open(list_file_path, 'wb') as list_file:

            for filename in files:
                if filename.endswith("jpg"):
                    file_path = os.path.join(root, filename)
                    img = image.load_img(file_path, target_size=(224,224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)

                    features = model.predict(x)

                    np_name = filename[0:-4]
                    np_name = np_name+".npy"
                    np.save(os.path.join(root,np_name), features)

        #            npy = open(os.path.join(root,np_name),"w+")
                    print('file %s (full path: %s)' % (filename, file_path))
                    list_file.write(('%s\n' % filename).encode('utf-8'))


start()


'''
    pic = out[:, :, 511]
    pylab.imshow(pic)
  #  pylab.gray()
    pylab.show()
    cpu = np.load('filecpu.npy')
    gpu = np.load('file.npy')
    model.summary()

'''