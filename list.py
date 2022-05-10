import os
import sys
import numpy
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
path="D:/12-14/INT16/rawDateset/val3_noShake"
filenames=os.listdir(path) #读取path内所有文件名返回列表
with open("D:/12-14/INT16/rawDateset/val3_noShake.txt","w") as f:
        for filename in filenames:
            sub_path=path+"/"+filename+"/"
            pic=os.listdir(sub_path)
            print(filename+','+str(filename))
            for p in pic:
                f.write('/'+filename+'/'+p+' '+str(filename)+'\n')

# with open ('F:/signal/CNN/torchproject/traindata/1/Zone_20200220103740567.txt','r') as f:
#     result = []
#     my_data = f.readlines() 
#     for line in my_data:
#         line = line.split()
#         line = line.replace(',',' ')
#         result.append(line)
# print(result)

# a=[1,2,3]
# b=[0,1,2]
# a.extend(b)
# print(a)

# a=[1,2,3]
# b=[3,2,1]
# a = numpy.array(a)
# b = numpy.array(b)
# c=sum(a==b) 
# print(c)
'''
画图
'''
# import matplotlib.pyplot as plt
# x1 = range(5)
# x2 = range(5)
# y1 = [1,2,3,4,5]
# y2 = [1,2,3,4,5]
# y3 = [2,4,6,8,10]
# y4 = [2,4,6,8,10]
# plt.subplot(2,1,1)
# plt.plot(x1, y1, 'o-',label = "train", color='b')
# plt.plot(x1,y3,'o-',label = "test" , color='r')
# plt.legend(loc='upper left')
# plt.title('accuracy vs. epochs')
# plt.ylabel('accuracy')
# plt.subplot(2,1,2)
# plt.plot(x2, y2, '.-',label = "train", color='b')
# plt.plot(x2,y4,'.-',label = "test", color='r')
# plt.legend(loc='upper left')
# plt.title('loss vs. epochs')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# #plt.savefig("accuracy_loss.jpg")
# plt.show()
'''
分类指标NAR
'''
# y_ture = [0,1,2,3,4,2,2,4,0]
# y_pre = [0,1,2,2,3,4,2,4,0]
# cm = confusion_matrix(y_ture,y_pre)
# print(cm)
# NAR = (sum(cm[0])-cm[0][0])/sum(cm[0])
# print(NAR)

#计算TCN层数
# r_in = 1
# k = 7
# for i in range(10):   
#     d = 2 ** i
#     r_out = r_in+(k-1)*d
#     r_out = r_out+(k-1)*d
#     print(r_out)
#     r_in = r_out

#fh = open('./val.txt','r')


# def _check_kernel_size_consistency(kernel_size):
#     if not (isinstance(kernel_size, tuple) or
#             (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
#         raise ValueError('`kernel_size` must be tuple or list of tuples')
# a = (3,3)
# a = tuple(a)
# b = (3,3)
# b = tuple(b)
# kernel_size = [a,b]
# _check_kernel_size_consistency(kernel_size)
