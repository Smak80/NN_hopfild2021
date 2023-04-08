import numpy as np
from PIL import Image
import math
class Kos:
    __size=(100,100) #Размер изображения

    def __init__(self, train_files, train_results):
        n=self.__size[0]*self.__size[1]
        self.__w=np.zeros((n,n))
        self.__kosko(train_files, train_results)

    def __mat2vec(self,x):
        '''Функция преобразования матрицы в вектор'''
        m = x.shape[0] * x.shape[1]
        c = np.reshape(x,(1,m))
        return c

    def __vectomat(self,x):
        m = int(math.sqrt(x.shape[1]))
        tmp1=np.reshape(x,(m,m))
        return tmp1

    def __create_W(self, x, y):
        '''Функуия нахождения весовой матрицы'''
        #if x.shape[0]!= 1:
        #    print("Введен не вектор")
        #    return
        #else:
        w = np.dot(x.T,y)
            #np.fill_diagonal(w,0)
        return w

    #картинку сделать одного размера 100 на 100.
    def __readImg2array(self, file, size, threshold= 145):
        '''Функуия преобразования картинки в массив '''
        pilIN = Image.open(file).convert(mode="L")#Черно-белой
        pilIN= pilIN.resize(size)
        imgArray = np.asarray(pilIN, dtype=np.uint8)#
        x = np.zeros(imgArray.shape, dtype=np.float)#возвращает новый массив заполненый нулями
        #у нас массив фоток из разных цифр делаем массив из 1 и -1
        x[imgArray > threshold] = 1
        x[x==0] = -1
        return x

    def __array2img(self, data, outFile = None):
        y = np.zeros(data.shape,dtype=np.uint8)
        y[data ==  1] = 255
        y[data == -1] = 0
        img = Image.fromarray(y, mode="L")
        return img

    def __sgn(self,x):
        if x > 0:
            res = 1
            return res
        elif x < 0:
            res = -1
            return res
        elif x == 0:
            res = 0
            return res

# есть ограничения есмкость сети,про это рассказать .На практике может запомнить больше но гарантировать нельзя
#
    def classify(self,test_file):
        x_er = self.__readImg2array(test_file, self.__size)
        y = self.__mat2vec(x_er)
        #z = self.__vectomat(y)

        r = np.sign(np.dot(y, self.__w))
        return r
        #k = -1
        # прервать процесс так ка может зациклить
        #t=y.T
        #while (k < 0):
        #    e = np.dot(self.__w, t)
        #    e = np.sign(e)
        #    if ((e==t).all()):
        #        k = 1
        #    else:
        #        t = e
        #z = self.__vectomat(t.T)
        # img=array2img(z)
        #img = self.__array2img(z)
        #img.show()

    def __kosko(self, train_files, train_results):
        x = np.array(
            [self.__mat2vec(self.__readImg2array(i, self.__size))[0] for i in train_files]
        )
        y = train_results
        self.__w = self.__create_W(x, y)
