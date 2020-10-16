# Импорт библиотек
import GAN
import PicLoad
import ImageGeneratorTF

# Задание размера изображений для работы сети
input_shape = (1024, 1024)
batch_size = 1
#help(PicLoad)
# Определение класса из модуля
picload = PicLoad.PICLOAD()

# Определение путей
path_input = r'D:\NeuroNet\GAN\dataSumWin\train_mask'
path_target = r'D:\NeuroNet\GAN\dataSumWin\train_img'

# Определение дополнительных параметров модуля
picload.basePath = [path_input, path_target]

# Загрузка данных
input = picload.load_img(picload.basePath[0], input_shape)
target = picload.load_img(picload.basePath[1], input_shape)

# Вывод примеров с датасетов
picload.show_imeges(input)
picload.show_imeges(target)

# Определение класса из модуля
imageGenerator = ImageGeneratorTF.IMAGEGENERATOR()

# Определение дополнительных параметров модуля
imageGenerator.img_size = input_shape
imageGenerator.batch_size = batch_size

# Запуск генератора изображений
train_generator = imageGenerator.conect_gen_image(
    imageGenerator.image_datagen_augmentation(path_input, input_shape, batch_size), 
    imageGenerator.image_datagen_augmentation(path_target, input_shape, batch_size))

# Определение сети GAN
gan = GAN.GAN()

# Определение дополнительных параметров модуля
gan.dataset = train_generator

# Запуск тренировки GAN
gan.train(gan.d_model, gan.g_model, gan.gan_model, gan.dataset)

