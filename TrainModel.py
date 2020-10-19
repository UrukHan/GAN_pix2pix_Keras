# Importing Libraries
import GAN
import PicLoad
import ImageGeneratorTF
import keras

# Setting image size for network operation
input_shape = (512, 512)
batch_size = 1

# Defining a class from a module
picload = PicLoad.PICLOAD()

# Defining paths
path_input = r'D:\NeuroNet\GAN\dataSumWin\train_mask'
path_target = r'D:\NeuroNet\GAN\dataSumWin\train_img'

# Defining additional module parameters
picload.basePath = [path_input, path_target]

# Loading data
input = picload.load_img(picload.basePath[0], input_shape)
target = picload.load_img(picload.basePath[1], input_shape)

# Output examples from datasets
picload.show_imeges(input)
picload.show_imeges(target)

# Defining a class from a module
imageGenerator = ImageGeneratorTF.IG()

# Defining additional module parameters
imageGenerator.img_size = input_shape
imageGenerator.batch_size = batch_size

# Run the image generator
train_generator = imageGenerator.conect_gen_image(
    imageGenerator.augmentation_wuth_std(path_input, input_shape, batch_size), 
    imageGenerator.augmentation_wuth_std(path_target, input_shape, batch_size))


# GAN network definition
gan = GAN.GAN()

# Load weights to continue training from the save location
gan.gan_model = keras.models.load_model(r'D:\NeuroNet\GAN\Models\ganModels')
gan.g_model = keras.models.load_model(r'D:\NeuroNet\GAN\Models\gModels')
gan.d_model = keras.models.load_model(r'D:\NeuroNet\GAN\Models\dModels')

# Defining additional module parameters
gan.dataset = train_generator

# Start GAN workout
gan.train(gan.d_model, gan.g_model, gan.gan_model, gan.dataset)

