# Импорт библиотек
import cv2
import os
from cv2 import cv2
from pylab import *

# Класс для работы с базой данных изображений
class PICLOAD():
    # Инициализация класса
    def __init__(self):
        pass

    # Функция возврата изображений через генератор
    def l_files(self, basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains = None):
        # Проход по структуре коталогов
        for (rootDir, dirNames, filenames) in os.walk(basePath):
            # Перебор имен файлов текущего каталога
            for filename in filenames:
                if contains is not None and filename.find(contains) == -1:
                    continue
                #  Определияем расширение текущего файла
                ext = filename[filename.rfind("."):].lower()

                # Проверка на изображение и обработку
                if ext.endswith(validExts):
                    # Возвращение изображения через генератор
                    imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                    yield imagePath

    # Возвращаем набор файлов директории и поддиректорий    
    def l_images(self, basePath, contains=None):
        return self.l_files(basePath, validExts = (".jpg", ".jpeg", ".png", ".bmp"), contains = contains)

    # Функция загрузки изображений          
    def load_img(self, basePath, img_size):
        images = []
        basePath = list(self.l_images(basePath))
        for path in basePath:
            if not('OSX' in path):
                path = path.replace('\\','/')
                image = cv2.imread(path) # Считывание изображений
                image = cv2.resize(image, img_size) # Изменение расширения изображений
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return images 
    
    # Функция отображения картинок в ряд
    def show_imeges(self, img):
        _,ax = plt.subplots(1, 3, figsize = (10, 30)) 
        for i in range(3):
            ax[i].imshow(img[i])
        return

    