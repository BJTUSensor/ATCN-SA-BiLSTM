#!/usr/bin/env python
# coding:utf8
import argparse
import logging
import os
import time
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler

#from models.cblstm import BLSTM,Regularization
from models.TCB import TCN
from dataset import DataFromFolder
import warnings
warnings.filterwarnings('ignore')

 
# 设备配置（如有GPU，则使用GPU进行加速）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
def draw(acc,allloss,vacc,vloss,NAR_list):
    x1 = range(len(acc))
    x2 = range(len(allloss))
    y1 = acc
    y2 = allloss
    y3 = vacc
    y4 = vloss
    y5 = NAR_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-',label = "train", color='b')
    plt.plot(x1,y3,'o-',label = "test" , color='r')
    plt.plot(x1,y5,'o-',label = "test_NAR" , color='g')
    plt.legend(loc='upper left')
    plt.title('accuracy & NAR vs. epochs')
    plt.ylabel('accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-',label = "train", color='b')
    plt.plot(x2,y4,'.-',label = "test", color='r')
    plt.legend(loc='upper left')
    plt.xlabel('loss vs. epochs')
    plt.ylabel('loss')
    plt.savefig("base_30.jpg")
    plt.show()
    


def test(model,dataset,args,criterion):
    """
    :param model:
    :param args:
    :param dataset:
    :param data_part:
    :return:
    """
    tic = time.time()
    model.eval()
    total_batch_num = 0
    total_num = 0
    prediction = []
    labels = []
    val_loss = 0

    ''' Prepare data and prediction'''
    total_features=numpy.empty([272,30],dtype=numpy.float64)

    #L1正则化
    #reg_loss=Regularization(model, weight_decay=1e-4, p=1)
    with torch.no_grad():
        for j,batch in enumerate(dataset):
            #计算总batch数量
            total_batch_num = total_batch_num+1
            #读数据
            vdata =Variable(batch['tdata'])
            batch_vdata = numpy.empty([args.batch_size,32,4000],dtype=numpy.float64)
            for i in range(args.batch_size):
                x = vdata[i,:,:]
                x = x.reshape(-1,1)
                ms = MinMaxScaler(feature_range=(0,255))
                x = ms.fit_transform(x)
                x = x.reshape(32,4000)
                batch_vdata[i,:,:] = x
            batch_vdata = torch.tensor(batch_vdata)
            vdata = batch_vdata
            vdata = vdata.float()
            vdata = vdata.to(device)
            vdata = vdata.view(args.batch_size * 32,1,4000)
            vlabel = Variable(batch['label'])
            save_label=vlabel.unsqueeze(1)
            #total_label[0+j*args.batch_size:8+j*args.batch_size,:]=save_label
            # print("label")
            # print(vlabel)
            vlabel = vlabel.to(device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            #计算结果
            #probs,features,safea = model(vdata)
            probs = model(vdata) 
            loss =  criterion(probs.view(len(vlabel),-1),vlabel)
            '''输出特征
            probs,features = model(vdata)
            features = features.reshape(args.batch_size, -1, 1)
            features = features.squeeze(2)
            total_features[0 + j * args.batch_size:args.batch_size + j * args.batch_size,:] = features.cpu().detach().numpy()
            '''
            loss =  criterion(probs.view(len(vlabel),-1),vlabel)
            #loss = loss + reg_loss(model)

            _, pred = torch.max(probs, dim=1)
            predi = pred.cpu()
            predi = numpy.asarray(predi.view(-1))  
            ##转成list，方便统计所有batch的分类结果 
            predi = predi.tolist()  
            label = vlabel.cpu()            
            label = label.tolist()

            val_loss += loss.item()
            
            #将pred中所有元素加入prediction中  
            prediction.extend(predi)#将pred中所有元素加入prediction中
            #pred = pred.view(-1).numpy()
            #统计所有label        
            labels.extend(label)
    
        tit = time.time()-tic
        #print(labels)
        #print(prediction)
        #计算metric,通过计算confusion_matirx得到NAR
        accuracy = accuracy_score(labels, prediction)
        c_matrix = confusion_matrix(labels,prediction)
        tNAR = (sum(c_matrix[0])-c_matrix[0][0])/sum(c_matrix[0])
        tFNR = (sum(c_matrix[:,0])-c_matrix[0][0])/sum(c_matrix[:,0])
        print(c_matrix)
    # precision = precision_score(labels, prediction, average='macro')
    # f1 = f1_score(labels, prediction, average='macro')
    # recall = recall_score(labels, prediction, average='macro')

    # print("  Predicting {:d} examples using {:5.4f} seconds\n".format(total_batch_num, tit))
    # print("Accuracy:",accuracy," Precision:", precision, " f1 score:", f1, " recall:", recall)
    # print(total_batch_num)
    # numpy.savetxt('label.csv', total_label, delimiter = ',')
    numpy.savetxt('tcnsafeatures.csv', total_features, delimiter = ',')
    # numpy.savetxt('SAfeatures.csv', total_safea, delimiter = ',')
    return accuracy,val_loss/total_batch_num,loss,tNAR,tFNR
    #return accuracy,tNAR
    

def train(model,train_x,train_y,args,optimizer,criterion):

    model.train()

    batch_size = args.batch_size
    
    #L1正则化
    #reg_loss=Regularization(model, weight_decay=1e-4, p=1)
    '''将数据转为tensor'''
    '''分出数据train_x和标签label'''
    

    model.zero_grad()
    probs = model(train_x)
    #print("111")
    #print(probs)

    
    loss = criterion(probs.view(len(train_y),-1),train_y)
    #loss = loss + reg_loss(model)


    _, pred = torch.max(probs, dim=1)
    #print(pred)
    
    #将GPU上运算的tensor复制到CPU上
    labels = train_y.cpu()
    #print("标签")
    #print(labels)

    labels = numpy.asarray(labels)
    labels = labels.tolist()
    
    predi = pred.cpu()
    predi = numpy.asarray(predi.view(-1))
    predi = predi.tolist() 

    #print(probs.view(len(train_y),-1).shape)
    #print(train_y.shape)
    
    loss.backward()
    optimizer.step()
    return labels,predi,loss.item()  #返回一个batch的准确数量和loss


def main(args):
    # define location to save the model
    if args.save == "__":
        args.save = "saved_model/%s_%s" % \
                    (args.model, "base_30")

    #划分训练集、测试集
    training_set = DataFromFolder(args.txtpath,args.root)
    print(len(training_set))
    train_data_loader = DataLoader(dataset=training_set,batch_size=args.batch_size,shuffle=True)
    val_set = DataFromFolder(args.txt2path,args.root2)
    val_data_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,shuffle=False)

    ##log日志
    logger = get_logger('./result/Atcnsafeaturetest.log')
    #dataset = ##load data
    #load training set && labelset
    '''测试代码'''
    is_test = False
    criterion = nn.CrossEntropyLoss()#原247行
    if is_test:
        with open(args.save + "/model.pt",'rb') as f:
            model = torch.load(f)
        acc_score,loss_score,val_loss,NAR,FNR=test(model, val_data_loader, args,criterion)
        print(NAR)
    else:


        '''训练'''
        if not os.path.exists(args.save):
            os.mkdir(args.save)
        channel_sizes = [args.hid] * args.levels
        #models = {"TBLSTM": c}
        model = TCN(batch_size=args.batch_size,
                                input_dim=args.hid,
                                hidden_dim=args.nhid,
                                output_dim=4,
                                input_size=args.input,
                                num_channels=channel_sizes,
                                kernel_size=args.ksize,
                                num_layers=args.nlayers,
                                dropout=args.dropout)
        model = model.to(device)
        #optimizer = optim.SGD(model.parameters(),lr=args.lr,weight_decay=1e-4,momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00008, weight_decay=1e-5)##Adam 包含对w、b的L2正则化
        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        #scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=1, verbose=True)#学习率的调整
        
        #画训练loss、accuracy迭代折线图
        acc_list = []
        loss_list = []
        vacc = []
        vloss = []
        vNAR = []
        vFNR = []


        best_acc_valid = 0.0#0.9999
        batches_per_epoch = int(len(training_set)/args.batch_size)
        max_train_steps = int(args.epochs * batches_per_epoch)
        for epoch in range(args.epochs):
            tic = time.time() 
            #每轮统计所有预测与标签值
            trainpredict = []
            trainlable = []

            runloss = 0
            #print("--------------------\nEpoch %d begins!" % epoch)
            for j,batch in enumerate(train_data_loader):    
                batch_x =Variable(batch['tdata'])
                '''for 归一化'''
                batch_data = numpy.empty([args.batch_size,32,4000],dtype=numpy.float64)
                for i in range(args.batch_size):
                    x = batch_x[i,:,:]
                    x = x.reshape(-1,1)
                    ms = MinMaxScaler(feature_range=(0,255))
                    x = ms.fit_transform(x)
                    x = x.reshape(32,4000)
                    batch_data[i,:,:] = x
                batch_data = torch.tensor(batch_data)
                batch_x = batch_data
                batch_x =batch_x.float()
                batch_x = batch_x.to(device)
                batch_x = batch_x.view(args.batch_size * 32,1,4000)
                batch_y = Variable(batch['label'])
                batch_y = batch_y.to(device)
                ####a是一个batch中分类准确的数量,l返回一个batch的loss
                tlabels,tpredi,tloss =train(model,batch_x,batch_y,args,optimizer,criterion)
                #计算一轮训练集所有batch的准确率和loss
                runloss = runloss+tloss
                trainlable.extend(tlabels)
                trainpredict.extend(tpredi)

            taccuracy = accuracy_score(trainlable, trainpredict)
            
            print("  using %.5f seconds" % (time.time() - tic))##训练当前轮次所花时间
            tic = time.time()  
            '''测试验证集'''
            print("Begin to predict the results on Validation")
            acc_score,loss_score,val_loss,NAR,FNR= test(model, val_data_loader, args,criterion)###验证集计算准确率  
                
            #统计一轮验证集与训练集分类表现
            vacc.append(acc_score)
            vloss.append(loss_score)
            vNAR.append(NAR)
            vFNR.append(FNR)
            
            acc_list.append(taccuracy)
            loss = runloss/batches_per_epoch
            loss_list.append(loss)

            #根据val_loss调整学习率
            #scheduler.step(val_loss)

            print("Epoch %d Train_accuracy %.3f Train_loss %.3f Val_accuracy %.3f Val_loss %.3f NAR %.3f FNR %.3f" %(epoch,taccuracy,loss,acc_score,loss_score,NAR,FNR))
            print("  Old best acc score on validation is %f" % best_acc_valid)
            if acc_score > best_acc_valid:
                print("  New acc score on validation is %f" % acc_score)
                best_acc_valid = acc_score
                with open(args.save + "/model.pt", 'wb') as to_save:
                    torch.save(model, to_save)
            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_loss={:.5f}\t train_acc={:.3f}\t val_acc={:.3f}\t NAR={:.3f}\t FNR={:.3f}'.format(epoch , args.epochs, loss, loss_score,taccuracy, acc_score, NAR,FNR))

        #画训练acc与loss迭代折线图
        draw(acc_list,loss_list,vacc,vloss,vNAR)          

        # with open(args.save + "/model.pt", 'wb') as to_save:
        #     torch.save(model, to_save)

    #print final result
    # with open(args.save + "/model.pt") as f:
    #     model = torch.load(f)
    #test(model, val_data_loader, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLSTM for classification")
    
    '''save model'''
    parser.add_argument("--save",type=str,default="__",
                        help="path to save model")

    '''model parameters'''
    parser.add_argument("--txtpath",type=str,default="../TCNBiLSTM/base_30.txt",
                        help="path of train_list")
    parser.add_argument("--txt2path",type=str,default="../TCNBiLSTM/val1.txt",
                        help="path of val_list")
    parser.add_argument("--root",type=str,default='../TCNBiLSTM/train1',
                        help="rootpath of traindata")
    parser.add_argument("--root2",type=str,default="../TCNBiLSTM/val1",
                        help="rootpath of valdata")                    
    parser.add_argument("--model",type=str,default="TBLSTM",
                        help="type of model to use for clasification")
    parser.add_argument("--input",type=int,default=1,
                        help="channels of input")
    parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
    parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 8)')               
    parser.add_argument("--nhid",type=int,default=15,
                        help="size of RNN hidden layer")#隐藏层状态数，自定义
    parser.add_argument("--nlayers",type=int,default=1,
                        help="number of layers")
    parser.add_argument('--hid', type=int, default=25,
                    help='output channel of TCN')
    parser.add_argument("--lr",type=float,default=0.01,
                        help="learning rate")
    parser.add_argument("--epochs",type=int,default=45,
                        help="number of training epochs")
    parser.add_argument("--batch_size",type=int,default=2,
                        help="batch size")
    parser.add_argument("--dropout",type=float,default=0.04,
                        help="dropout rate")
    parser.add_argument("--is_test", action="store_true",
                        help="flag for training model or only test")

    my_args = parser.parse_args()

    main(my_args)
