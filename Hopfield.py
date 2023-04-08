import numpy as np
from PIL import Image
import math
class Hop():
#Можно в конструктор добавить файл,в который будет сохраняться картинка
   # '''
   # Эта функция демонстрирует использование docstring в Python.
   # '''

    __size=(100,100)
    '''Размер изображения'''
    def __init__(self,train_files):
        n=self.__size[0]*self.__size[1]
        self.__w=np.zeros((n,n))
        self.__hopfield(train_files)

    def __mat2vec(self,x):
        '''Функция преобразования матрицы в вектор'''
        m = x.shape[0] * x.shape[1]
        # tmp1 = np.zeros(m)
        # c = 0
        #
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         tmp1[c] = x[i, j]
        #         c += 1
        c=np.reshape(x,(1,m))
        return c
    def __vectomat(self,x):
        '''Функуия преобразования вектора в матрицу'''
        # tmp1=np.zeros(self.__size)
        # c = 0
        # m=math.sqrt(len(x))
        # for i in range(100):
        #     for j in range(100):
        #         tmp1[i,j]=x[c]
        #         c +=1
        #
        m = int(math.sqrt(x.shape[1]))
        tmp1=np.reshape(x,(m,m))
        return tmp1
    def __create_W(self,x):
        '''Функуия нахождения весовой матрицы'''
        if x.shape[0]!= 1:
            print("Введен не вектор")
            return
        else:
            #w = np.zeros([len(x),len(x)])
            # for i in range(len(x)):
            #     for j in range(i,len(x)):
            #         if i == j:
            #             w[i,j] = 0
            #         else:
            #             w[i,j] = x[i]*x[j]
            #             w[j,i] = w[i,j]
            w = np.dot(x.T,x)
            #w.fill_diagonal(0)
            np.fill_diagonal(w,0)
        return w
#картинку сделать одного размера 100 на 100.
    def __readImg2array(self,file,size, threshold= 145):
        '''Функуия преобразования картинки в массив '''
        pilIN = Image.open(file).convert(mode="L")#Черно-белой
        pilIN= pilIN.resize(size)
        #pilIN.thumbnail(size,Image.ANTIALIAS)
        imgArray = np.asarray(pilIN,dtype=np.uint8)#
        x = np.zeros(imgArray.shape,dtype=np.float)#возвращает новый массив заполненый нулями
       #у нас массив фоток из разных цифр делаем массив из 1 и -1
        x[imgArray > threshold] = 1
        x[x==0] = -1
        return x
    def __array2img(self,data, outFile = None):
        '''Функуия преобразования массива в картинку'''
        #data is 1 or -1 matrix
        y = np.zeros(data.shape,dtype=np.uint8)
        y[data==1] = 255
        y[data==-1] = 0
        img = Image.fromarray(y,mode="L")
        # if outFile is not None:
        #     img.save(outFile)
        return img
    def __sgn(self,x):
        #np.sign(x)
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
        z = self.__vectomat(y)
        #q = np.dot(self.__w, y)
        #e = np.zeros(q.shape)
        k = -1
        # прервать процесс так ка может зациклить
        t=y.T
        while (k < 0):
            #t = np.sign(q)
            e = np.dot(self.__w, t)
            e = np.sign(e)
            # тут
            # for i in range(len(y)):
            #     t[i] = self.__sgn(t[i])
            # e = np.dot(self.__w, t)
            if ((e==t).all()):  # np.allclose
                # all,any для обхода массива
                k = 1
            else:
                t = e
        z = self.__vectomat(t.T)
        # img=array2img(z)
        img = self.__array2img(z)
        img.show()
    def __hopfield(self,train_files):
        # r = self.__readImg2array(train_files[0], size)
        # x = self.__mat2vec(r)
        # w = self.__create_W(x)
        #
        # file2 = "C:\\Users\\артур\\Desktop\\hop\\len.png"
        # rr = self.__readImg2array(train_files[1], size)
        # xx = self.__mat2vec(rr)
        # ww = self.__create_W(xx)
        # w=w+ww
        #w = np.zeros((10000, 10000))
        for i in train_files:
            r = self.__readImg2array(i, self.__size)
            x = self.__mat2vec(r)
            tmp_w = self.__create_W(x)
            self.__w =self.__w+tmp_w

