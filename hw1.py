# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:45:47 2024

@author: user
"""

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#%%
myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
#%%
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False,
                 select_feature_name=[], ):
        self.mode = mode
        '''数据预处理'''
        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data_column = np.array(data[0])[1:]
            data = np.array(data[1:])[:, 1:].astype(float) #np.array(data[1:])需要第一行的，[:, 1:]需要第一列后面的
            
            
        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            if select_feature_name==[] :
                print('input the feature name as list')
            else:
                #f_index = np.where(np.isin(np.array(select_feature_name),data_column)) 
                feats=list(range(40))
                for i in select_feature_name:
                    k=0
                    for j in data_column:
                        if i == j :
                            print(k)
                            feats.append(k)
                        k=k+1
                if mode=='test':
                    feats = feats
                else:
                    
                   feats=feats[0:-1]
                print(feats)
                # index = list( set(list((range(93)))) - set(f_index[0].tolist()))
                # feats = f_index[0].tolist()
                #data.take(index,1) = 0#间隔提取列，1表示列，0是行
            # pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)

        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            # target = data[:, -1] #准确的结果,即label
            target = data[:, feats[-1]] #准确的结果,即label
            data = data[:, feats] 

            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0] #返回余数，不需要10的倍数
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            #print(indices)#tran+dev一共2700个sample。#???
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        '''
        feature scaling:为了消除特征间单位和尺度差异的影响，以对每维特征同等看待，需要对特征进行归一化。
        原始特征下特征下，因尺度差异，其损失函数的等高线图可能是椭圆形
        梯度下降算法需要feature scaling，
        zero center与参数初始化相配合，缩短初始参数位置与local minimum间的距离，加快收敛。
        不同方向上的下降速度变化不同（二阶导不同，曲率不同），feature scaling改变了损失函数的形状，减小不同方向上的曲率差异。
        scaling后不同方向上的曲率相对更接近，更容易选择到合适的学习率，使下降过程相对更稳定。
        对于传统的神经网络，对输入做feature scaling也很重要，因为采用sigmoid等有饱和区的激活函数，
        如果输入分布范围很广，参数初始化时没有适配好，很容易直接陷入饱和区，导致梯度消失，
        '''
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
    
#%%    
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False,select_feature_name=[]):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only,select_feature_name=select_feature_name)  # Construct dataset
    dataloader = DataLoader(       # Construct dataloader
        dataset, batch_size,
        shuffle=(mode == 'train'), # shuffle->training=true，下次读取打乱顺序,testing=false，不打乱顺序
        drop_last=False, #样本数不能被batch_size整除时，最后一批数据是否舍弃（default：False）
        num_workers=n_jobs, #是否是多进程读取数据，默认为0
        pin_memory=True) #如果为True会放到GPU上，默认为False'            
    return dataloader
#%%
#
'''定义model'''
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), #input_dim, n_hidden1
            nn.ReLU(),
            nn.Linear(64, 1) #n_hidden1,output_dim
        )
        '''
        模型在训练集上表现很差，说明模型高偏差，此时模型欠拟合
        模型在验证集上差，在测试集上很好，说明模型高方差，此时过拟合
        '''

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''#??? 没找到x
        return self.net(x).squeeze(1)


    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)

#%%
''' 训练模型'''
def train(tr_set, dv_set, model, config, device):
    ''' DNN training
    config：一些参数
    '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record
#%%
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss     
#%%
def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds
#%%     
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 256,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}
     
#%%
# COVID19Dataset(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.train.csv', mode='train', target_only=True,select_feature_name=['worried_finances',],)
#COVID19Dataset(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.train.csv', mode='dev', target_only=False)
#%%
tr_set = prep_dataloader(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.train.csv', 'train', config['batch_size'], target_only=1,select_feature_name=['tested_positive',])
dv_set = prep_dataloader(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.train.csv', 'dev', config['batch_size'], target_only=1,select_feature_name=['tested_positive',])
tt_set = prep_dataloader(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.test.csv', 'test', config['batch_size'], target_only=1,select_feature_name=['tested_positive',])
# tr_set = prep_dataloader(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.train.csv', 'train', config['batch_size'], target_only=target_only,select_feature_name=['tested_positive',])
# dv_set = prep_dataloader(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.train.csv', 'dev', config['batch_size'], target_only=target_only,select_feature_name=['tested_positive',])
# tt_set = prep_dataloader(r'C:\Users\user\Desktop\ml2021spring-hw1\covid.test.csv', 'test', config['batch_size'], target_only=target_only,select_feature_name=['tested_positive',])
#%%
model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device    #tr_set.dataset.dim应该就是feature的数量  

#%%
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
#%%
model_loss

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
    #%%
plot_learning_curve(model_loss_record, title='deep model')
#%%
del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set
#%%
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
#save_pred(preds, r'C:\Users\user\Desktop\ml2021spring-hw1\pred.csv')         # save prediction file to pred.csv

#%%
