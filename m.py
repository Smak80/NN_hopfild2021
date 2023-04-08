import numpy as np
from PIL import Image
import Hopfield
#Не исправлено работа сразу с несколькими картинками
#Емкость сети-сколько образцов может распознать при
#данном количестве нейронов
# нейроны -длина поступаемого вектора(У НАС ФОТКА 100*100 длина вектора 10000=N
#образцы-наши файлы с фотографиями(3 фотки) =M
# M=N/log(2)N- по Хайкину. В других источниках Ln N
#  нашем случае M=10000/log(2)10000=10000ln(2)/ln(10000)=752
#Может запомнить 752 фотки
import Kosko


def toNumber(x):
    x[x == -1] = 0
    r = 0
    #print(x[0])
    for i in range(len(x[0])):
        r += (1 << i) * x[0][len(x[0])-i-1]
        #print(i, 1 << i, x[0][len(x[0])-i-1], r)
    return int(r)

file1="img\\len.png"
file2="img\\men.png"
file3="img\\merl.png"

file1Er="img\\lenEr.png"
file2Er="img\\menEr.png"
file3Er="img\\merl.Er.png"
file4Er = "img\\4.png"
file5Er = "img\\5.png"
train_files=[file1, file2, file3]
train_results = [[-1,-1, -1], [-1, -1, 1], [-1, 1, -1]]
result_labels = ['Ленин', 'Мужчина', 'Монро']
#h=Hopfield.Hop(train_files)
#h.classify(file1Er)
k = Kosko.Kos(train_files, train_results)
res = toNumber(k.classify(file3Er))
lRes = result_labels[res] if res in range(len(result_labels)) else 'неизвестная картинка :('
print('Это', lRes)
# x=[1,2,3,4,5]
#
# r=0
# for i in range(0,len(x)):
#     r=r+x[i]
# print(r)

# one=np.array([[170,1,1,-1,1,1,1],
#      [400,1,1,-1,1,180,1]])
# x=np.zeros((2,7))
# print(x)
# x[one > 145] = 1
# print(x)
# x[x == 0] = -1
# print(x)
# print(one>145)
