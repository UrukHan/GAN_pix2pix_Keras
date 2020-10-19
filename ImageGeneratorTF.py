# Importing Libraries
from keras.preprocessing.image import ImageDataGenerator

# TensorFlow image generator class (input data:batch_size, img_size)
class IG():
    # Class initialization
    def __init__(self):
        self.seed = 1

    # Generator function to increase the number of images with STD(input data: path, img_size, batch_size)
    def augmentation_wuth_std(self, path, img_size, batch_size):
        # Setting parameters for augmentation
        image_datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center = True,
            featurewise_std_normalization = True,
            rotation_range = 10,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'reflect')
        # Generator definition
        image_generator = image_datagen.flow_from_directory(path,
                                                     target_size = img_size,
                                                     class_mode='categorical',
                                                     batch_size = batch_size,
                                                     interpolation = 'nearest',
                                                     seed = self.seed)
        return image_generator

    # Function for combining two input images generated by augmentation into an array
    # input data: image generators.
    def conect_gen_image(self, imageDatagen_I, imageDatagen_II):
        conect_gen_image = [imageDatagen_I, imageDatagen_II]
        return conect_gen_image


    # Generator function to increase the number of images no STD(input data: path, img_size, batch_size)
    def augmentation_no_std(self, path, img_size, batch_size):
        # Setting parameters for augmentation
        image_datagen = ImageDataGenerator(
            rotation_range = 10,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'reflect')
        # Generator definition
        image_generator = image_datagen.flow_from_directory(path,
                                                     target_size = img_size,
                                                     class_mode='categorical',
                                                     batch_size = batch_size,
                                                     color_mode = "grayscale",
                                                     interpolation = 'nearest',
                                                     seed = self.seed)
        return image_generator

    # Function for combining two input images generated by augmentation into an zip
    # input data: image generators.
    def zip_gen_image(self, imageDatagen_I, imageDatagen_II):
        zip_gen_image = [imageDatagen_I, imageDatagen_II]
        return zip_gen_image
        
