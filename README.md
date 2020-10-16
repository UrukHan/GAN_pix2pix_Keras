# GAN_pix2pix_Keras
GAN neural network. For converting images. Collected on Keras
I trained this network on 16 photos in total. Therefore, we had to fit the model and run the generator of augmented images on the keras.

GAN.py - a module in which the architecture of the GAN network is written, consisting of two neural networks. Discriminator and generator. Everything is collected in the "GAN" class. In this case, for processing 1024x1024 images. Also "GAN.train" and "GAN.generate_fake_samples" network training method.

PicLoad.py - a module that contains the "PICLOAD" class for viewing and loading images. In this class, you need to set the third-party parameter "basePath" indicating the path to the folder. the "load_img" method takes the path to the folder and the size of the images in this case, we make all the images 1024/1024 for our network.
There is also a "show_imeges" method for displaying three images on plot.

ImageGeneratorTF.py - module for image augmentation. It is itself a generator. I prescribed it for our network so that it gave out two images. At the testing stage, the second image is just a copy of the first just to keep the format. at the testing stage, when generating a picture by the generator, the weights remain as they were. This generator is represented by the "IMAGEGENERATOR ()" class. It requires two parameters to be set from outside. this is the batch size and image resolution. imageGenerator.img_size = input_shape
imageGenerator.batch_size = batch_size. Internally, the "conect_gen_image" method is set, which will concatenate two generated images by input. and also set the seed so that the transformations match, so the image connector is inside the class. This method accepts two other "image_datagen_augmentation" methods in which the path is specified the size of the batch (we have very few pictures everywhere for training) and the size of the images to which the initial ones need to be transformed (path_input, input_shape, batch_size).

TrainModel.py - file for training. In principle, everything that called classes do has already been described) Here we call modules. We define instances of classes, introduce third-party parameters to them and start the workout and wait. Everything is now wall-mounted so that every 25 epochs the result is saved in the form of three photos (original, predicted, target) the result is saved every 25 epochs in the "result" folder. The "save" parameter changes the save frequency in the GAN module in the train method. Saving model weights to the Models folder is also there. Also, on our training page, commands for loading synchronized models are booked in order to continue training from this place

Predict.py - file for generating predicted images. There are two functions at the beginning. "denorm_img" - image denormalization. since image matrices go into float with a range of values 0-1. And "gen_img" which normalizes the image, feeds the network for generation and denormalizes back at the output. This file also includes loading weights and saving predicted images to the "Predict" folder

Folders:
Models - to save weights of models
result - display of results during training
dataSumWin - Folder with a set of images. There are only 16 of them)) but the network has learned. There was no time to collect hundreds of pictures. The hierarchy is so important for the Keras generator. she looks for folders in the specified folder (this function was originally written for the classifier)


