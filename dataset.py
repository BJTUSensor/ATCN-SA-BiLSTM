from os.path import join
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd


class DataFromFolder(data.Dataset):
    def __init__(self, txt_path,dataroot):
        self.root = dataroot
        fh = open(txt_path,'r')
        Data = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            Data.append((words[0], int(words[1])))
            self.Data = Data
    def __getitem__(self, index):
        fn, label = self.Data[index]
        #print(fn)
        #print(label)
        a=self.root+fn
        #print(a)
        #tdata = np.loadtxt(a,delimiter='\t')
        tdata = pd.read_csv(a, header=None) 
        tdata = np.array(tdata)      
        #print(tdata)
        return {'tdata':tdata, 'label':label}
    def __len__(self):
        return len(self.Data)

# txt_path = "F:\signal\CNN\\torchproject\\train.txt"
# root = "\\traindata"
# train_set = DataFromFolder(txt_path,root)
# print(train_set)






