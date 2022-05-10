import torch.nn.functional as F
from torch import nn
import torch
from models.tcn import TemporalConvNet
from torch.autograd import Variable


class TCN(nn.Module):
    def __init__(self, batch_size,input_dim,hidden_dim,output_dim,input_size, num_channels, kernel_size, num_layers,dropout):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.batch_size = batch_size
        self.input_dim = num_channels[-1]  ##由卷积后输出的维度大小决定
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.outchannel = num_channels[-1]
        self.sen_len = 32
        '''att'''
        self.fc1 = nn.Linear(self.outchannel,5*3) 
        self.fc2 = nn.Linear(5*3,self.outchannel)
        '''Conv attention on seqlen'''
        self.attconv = nn.Conv1d(2,1,kernel_size=7,padding=3,stride=1)

        self.sen_rnn = nn.LSTM(input_size=input_dim, 
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=0.4,
                               batch_first=True,
                               bidirectional=True)
        
        self.output = nn.Linear(2 * self.hidden_dim, output_dim)
        
        #self.linear = nn.Linear(num_channels[-1], output_size)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        ###GPU运行.cuda()
        #fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())#按第三维度索引rnn_outs第一
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])))
        fw_out = fw_out.view(batch_size * max_len, -1)

        #bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())#按第三维度索引rnn_outs第二
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])))        
        bw_out = bw_out.view(batch_size * max_len, -1)

        #batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_range = Variable(torch.LongTensor(range(batch_size)))* max_len

        #batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()
        batch_zeros = Variable(torch.zeros(batch_size).long())

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)   
        y2 = y1[:,:,-1]
        '''channel attention
        att = F.relu(self.fc1(y2))
        att = F.softmax(self.fc2(att))
        #print(att)
        wf = y2*att   #.view(self.batch_size, -1))
        '''

        #o = self.linear(y1[:, :, -1])
        sen_batch = y2.unsqueeze(1)
        sen_batch = sen_batch.view(self.batch_size,-1,self.input_dim)
        sen_batch = sen_batch.permute(0,2,1)
 
        '''conv attention'''
        avg_out = torch.mean(sen_batch, dim=1, keepdim=True)
        max_out, _ = torch.max(sen_batch, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = F.sigmoid(self.attconv(x))
        f = x*sen_batch
        f = f.permute(0,2,1)
        ''' Bi-LSTM Computation '''
        sen_outs, _ = self.sen_rnn(f)
        sen_rnn = sen_outs.contiguous().view(self.batch_size, -1, 2 * self.hidden_dim)
        ''' Fetch the truly last hidden layer of both sides
        '''
        sen_lengths = [32,32]
        #sen_lengths = [32]
        sen_lengths = torch.tensor(sen_lengths)
        #sen_lengths =sen_lengths.cuda()
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, self.batch_size, self.sen_len)  # (batch_size, 2*hid)
        out = self.output(sentence_batch)
        # return out_probs, sentence_batch
        return out