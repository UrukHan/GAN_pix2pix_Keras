# Importing Libraries
import keras
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import io
from keras.models import Model
from keras.models import Input
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from random import randint


# Class GAN neural network of pix2pix type. Input data: dataset
class GAN():
    # Class initialization
    def __init__(self):
        self.image_shape = (512, 512, 3)
        # Defining discriminator, generator and GAN models
        self.d_model = self.define_discriminator()
        self.g_model = self.define_generator()
        self.gan_model = self.define_gan(self.g_model, self.d_model, self.image_shape)
        # Outputting the structure of the final network
        self.gan_model.summary()
        # Print model structure
        #plot_model(self.gan_model, to_file='gan_model_plot.png', show_shapes=True, show_layer_names=True)

    # Denormalize the image. (img, 0.0, 1.0, 0, 255) .astype ('uint8')
    def denorm_img(self, image, from_min, from_max, to_min, to_max):
        from_range = from_max - from_min
        to_range = to_max - to_min
        scaled = np.array((image - from_min) / float(from_range), dtype=float)
        return to_min + (scaled * to_range)

    # Function of creating discriminator model
    def define_discriminator(self):
        # Initializing the balance
        init = RandomNormal(stddev=0.02)
        # Determining the dimension of the input of two images
        in_src_image = Input(shape = self.image_shape)
        in_target_image = Input(shape = self.image_shape)
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = Conv2D(64, (4, 4), strides=(2, 2), padding = 'same', kernel_initializer = init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C1024
        d = Conv2D(1024, (4,4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # Penultimate layer
        d = Conv2D(2048, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # Model definition
        model = Model([in_src_image, in_target_image], patch_out)
        # Compilation
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model

    # Function to create a generator model
    def define_generator(self):

        # Create function
        def define_encoder_block(layer_in, n_filters, batchnorm=True):
            # Initializing the balance
            init = RandomNormal(stddev = 0.02)
            # Downsampling layer
            e = Conv2D(n_filters, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = init)(layer_in)
            # Adding batch normalization on condition
            if batchnorm:
                e = BatchNormalization(momentum = 0.9, epsilon = 0.00001)(e, training = True)
            # Activation function
            e = LeakyReLU(alpha = 0.2)(e)
            return e

        # Decoder creation function
        def decoder_block(layer_in, skip_in, n_filters, dropout=True):
            # Initializing the balance
            init = RandomNormal(stddev = 0.02)
            # Upsampling layer
            e = Conv2DTranspose(n_filters, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = init)(layer_in)
            # Adding batch normalization
            e = BatchNormalization()(e, training=True)
            # Adding Dropout provided
            if dropout:
                e = Dropout(0.5)(e, training=True)
            # Connecting to a connector
            e = Concatenate()([e, skip_in])
            # Activation function
            e = Activation('relu')(e)
            return e

        # Input dimension
        in_image = Input(shape = self.image_shape)
        # Initializing the balance
        init = RandomNormal(stddev=0.02)
        # Encoder model: C64-C128-C256-C512-C1024-C1024-C2048
        e1 = define_encoder_block(in_image, 64, batchnorm=False)
        e2 = define_encoder_block(e1, 128)
        e3 = define_encoder_block(e2, 256)
        e4 = define_encoder_block(e3, 512)
        e5 = define_encoder_block(e4, 1024)
        e6 = define_encoder_block(e5, 1024)
        e7 = define_encoder_block(e6, 2048)
        # Narrowed center of U network
        b = Conv2D(2048, (4, 4), strides = (2, 2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        #Decoder model: CD2048-CD1024-C512-C512-C256-C128-C64
        d2 = decoder_block(b, e7, 2048)
        d3 = decoder_block(d2, e6, 1024)
        d4 = decoder_block(d3, e5, 512)
        d5 = decoder_block(d4, e4, 512, dropout=False)
        d6 = decoder_block(d5, e3, 256, dropout=False)
        d7 = decoder_block(d6, e2, 128, dropout=False)
        d8 = decoder_block(d7, e1, 64, dropout=False)
        # Output
        g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d8)
        out_image = Activation('tanh')(g)
        # Model definition
        model = Model(in_image, out_image)
        opt = Adam(lr=0.0002, beta_1 = 0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model

    # Combining generator and discriminator neural networks in a GAN network model
    def define_gan(self, g_model, d_model, image_shape):
        # Disable learning (changing weights) of the discriminator
        d_model.trainable = False
        # Dimensions of processed images
        in_src = Input(shape = image_shape)
        # Connecting the input image and the output of the generator to the inputs of the discriminator
        gen_out = g_model(in_src)
        dis_out = d_model([in_src, gen_out])
        # Model definition. Input image, classification output and generated image
        model = Model(in_src, [dis_out, gen_out])
        # Compilation
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss = ['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])  # 1, 100
        return model

    # Pick a batch of random samples, return the images and pick class 1
    def generate_real_samples(self, dataset, n_samples, patch_shape):
        # Unpacking the dataset
        trainA, trainB = dataset
        # Choosing a random pair
        ix = randint(0, len(trainA)-1)
        X1, X2 = trainA[ix][0], trainB[ix][0]
        # generating real data with class 1
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y 

    # Generate batch of images, returns target image and class 0
    def generate_fake_samples(self, g_model, samples, patch_shape):
        # Generation of predicted images with class 0
        X = g_model.predict(samples)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    # Training function (Accepts the functions of a discriminator, generator and GAN. As well as a dataset specified from outside)
    def train(self, d_model, g_model, gan_model, dataset, save = 100, n_epochs = 1000, n_batch = 1, n_patch = 16):
        # Unpacking the dataset
        trainA, trainB = dataset
        # calculation of the number of parties for the training era
        bat_per_epo = int(len(trainA) / n_batch)
        # Calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # Recalculation of eras
        for i in range(n_steps):
            # Selection of batch of real images
            [X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
            # Create batch of generated images
            X_fakeB, y_fake = self.generate_fake_samples(g_model, X_realA, n_patch)
            # Discriminator updates for real data
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # Updates to the discriminator for generated data
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # Updating the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # Outputting the result
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            # Output of results and saving
            if i % save == 0 and i != 0 or i == 1:
                #cv2.imwrite('/content/results', X_fakeB)
                # print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss)) 
                img = self.denorm_img(X_fakeB[0], 0.0, 1.0, 0, 255).astype('uint8')
                fig, ax = plt.subplots(1, 3, figsize = (9, 3)) 
                ax[0].imshow(self.denorm_img(X_realA[0], 0.0, 1.0, 0, 255).astype('uint8'))
                ax[0].axis('off')
                ax[1].imshow(img)
                ax[1].axis('off')
                ax[2].imshow(self.denorm_img(X_realB[0], 0.0, 1.0, 0, 255).astype('uint8'))
                ax[2].axis('off')
                fig.savefig(r"D:\NeuroNet\GAN\result\%d.jpg" % (i)) 
                #plt.show()
            # Save model weights
            if i % save == 0 and i != 0:
                gan_model.save(r"D:\NeuroNet\GAN\Models\ganModels")
                g_model.save(r"D:\NeuroNet\GAN\Models\gModels")
                d_model.save(r"D:\NeuroNet\GAN\Models\dModels")
    


