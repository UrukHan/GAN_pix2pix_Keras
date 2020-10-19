# Importing Libraries
import GAN
import PicLoad
import ImageGeneratorTF
import cv2
import tensorflow as tf
import numpy as np
import keras
import os

# Function of generating an image for a given network
def gen_img(img):
  in_img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  input_test_img = tf.expand_dims(in_img, 0)
  gen_img = gan.generate_fake_samples(gan.g_model, input_test_img, 1)
  gen_img = (denorm_img(gen_img[0][0], 0.0, 1.0, 0, 255).astype('uint8'))
  return gen_img

# Denormalize the image. (img, 0.0, 1.0, 0, 255) .astype ('uint8')
def denorm_img(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


# Setting image size for network operation
input_shape = (1024, 1024)
batch_size = 1

# Path definition
path_test = r'D:\NeuroNet\GAN\dataSumWin\test\winter'

# Defining a class from a module
picload = PicLoad.PICLOAD()

# Defining additional module parameters
picload.basePath = path_test 

# Loading data
images_test = picload.load_img(picload.basePath, input_shape)

# GAN network definition
gan = GAN.GAN()

gan.gan_model = keras.models.load_model(r'D:\NeuroNet\GAN\Models\ganModels')
gan.g_model = keras.models.load_model(r'D:\NeuroNet\GAN\Models\gModels')
gan.d_model = keras.models.load_model(r'D:\NeuroNet\GAN\Models\dModels')


for i in range(len(images_test)):
    img = images_test[i].astype(float)
    img = gen_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(r"D:\NeuroNet\GAN\Predict" , r"%d.jpg" % (i)), img)

    
