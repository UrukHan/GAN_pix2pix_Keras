from keras.preprocessing.image import ImageDataGenerator


# Класс генератора изображений TensorFlow (входные данные: batch_size, img_size)
class IMAGEGENERATOR():
    # Инициализация класса
    def __init__(self):
        self.seed = 1

    # Функция генератора аугментации изображений (входные данные: path, img_size, batch_size)
    def image_datagen_augmentation(self, path, img_size, batch_size):
        # Запускае генерацию аугментированных изображений
        image_datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center = True,
            featurewise_std_normalization = True,
            rotation_range = 10,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'reflect')
        # Определение генератора
        image_generator = image_datagen.flow_from_directory(path,
                                                     target_size = img_size,
                                                     class_mode='categorical',
                                                     batch_size = batch_size,
                                                     interpolation = 'nearest',
                                                     seed = self.seed)
        return image_generator

    '''Функция объеденения двух входных, генерируемых аугментацией, избражений в массив
       входные данные: генераторы изображений'''
    def conect_gen_image(self, imageDatagen_I, imageDatagen_II):
        conect_gen_image = [imageDatagen_I, imageDatagen_II]
        return conect_gen_image

    # Функция генератора тестовых изображений (входные данные: path, img_size, batch_size)
    def image_datagen(self, path, img_size, batch_size):
        # Запускае генерацию аугментированных изображений
        image_datagen = ImageDataGenerator( 
            rescale=1. / 255,
            featurewise_center = True,
            featurewise_std_normalization = True,)
        # Определение генератора
        image_test = image_datagen.flow_from_directory(path,
                                                     target_size = img_size,
                                                     class_mode='categorical',
                                                     batch_size = batch_size,
                                                     interpolation = 'nearest',
                                                     seed = self.seed)
        return image_test
