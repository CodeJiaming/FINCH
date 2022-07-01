import torch.nn as nn
import torch
import re
import sys
import time
import numpy as np
import torch.nn.functional as F
import gc
from utils import printer, time_since
sys.path.append('..')
from nas_module.darts.architect import Architect

def MakeLayers(params_list):
    conv_layers = nn.Sequential()
    fc_layers = nn.Sequential()
    c_idx=p_idx=f_idx=1
    for param in params_list:
        if param[1] == 'Conv':
            conv_layers.add_module(param[0], nn.Conv2d(
                param[2][0], param[2][1], param[2][2], param[2][3],param[2][4]))
            if len(param) >=4:
                if param[3] == 'Batchnorm':
                    conv_layers.add_module(
                        'batchnorm'+str(c_idx), nn.BatchNorm2d(param[2][1]))
                if param[3]=='Relu' or (param[3] == 'Batchnorm' and param[4]=='Relu'):
                    conv_layers.add_module('relu'+str(c_idx),nn.ReLU(inplace=True))
                else:
                    conv_layers.add_module('sigmoid'+str(c_idx),nn.Sigmoid())
            if len(param) >=6 :
                if param[3] == 'Batchnorm':
                    if param[5]=='Maxpool':
                        conv_layers.add_module(
                            'maxpool'+str(p_idx), nn.MaxPool2d(param[6][0], param[6][1],param[6][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool'+str(p_idx), nn.AvgPool2dparam[6][0], param[6][1],param[6][2])
                else:
                    if param[4]=='Maxpool':
                        conv_layers.add_module(
                            'maxpool'+str(p_idx), nn.MaxPool2d(param[5][0], param[5][1],param[5][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool'+str(p_idx), nn.AvgPool2dparam[5][0], param[5][1],param[5][2])
                p_idx+=1
            c_idx+=1
            
        else:
            fc_layers.add_module(param[0], nn.Linear(param[2][0], param[2][1]))
            if len(param) >= 4:
                if param[3] == 'Dropout':
                    fc_layers.add_module('dropout', nn.Dropout(param[4]))
                if param[3] == 'Relu' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Relu'):
                    fc_layers.add_module('relu'+str(f_idx), nn.ReLU(inplace=True))
                elif param[3] == 'Sigmoid' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Sigmoid'):
                    fc_layers.add_module('sigmoid'+str(f_idx), nn.Sigmoid())
                elif param[3] == 'Softmax' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Softmax'):
                    fc_layers.add_module('softmax'+str(f_idx), nn.Softmax())
            f_idx+= 1
    return conv_layers, fc_layers


class MyNet(nn.Module):
    def __init__(self, params_list):
        super(MyNet, self).__init__()
        self.features, self.classifier = MakeLayers(params_list)

    def forward(self, x):
        if len(self.features) > 0:
            feature = self.features(x)
            linear_input = torch.flatten(feature, 1)
            output = self.classifier(linear_input)
        else:
            output = self.classifier(x)
        return F.log_softmax(output, dim=1)


#初始化模型的参数由以下几部分构成：
# 卷积层(每一层的名字,Conv,参数列表( , , , , ),Batchnorm(可选),激活函数(Relu,Sigmoid)，池化层(Maxpool,Avgpool,可选),参数列表( , , ))
#全连接层(每一层的名字,FC,参数列表( , ),Dropout(可选）,激活函数(Relu,Sigmoid,Softmax,可选))


LeNet5 = [('conv1', 'Conv', (1, 6, 5, 1,1), 'Sigmoid', 'Maxpool',(2,2,0)), ('conv2', 'Conv', (6, 15, 5, 1,0),'Sigmoid', 'Maxpool',(2,2,0)), 
             ('fc1', 'FC', (16*4*4, 120), 'Dropout',0.2,'Sigmoid'), ('fc2', 'FC', (120, 84), 'Sigmoid'),('fc3', 'FC', (84, 10))]


AlexNet=[('conv1', 'Conv', (3, 64, 11, 4,2), 'Relu', 'Maxpool',(3,2,0)), ('conv2', 'Conv', (64, 192, 5, 1,2),'Relu', 'Maxpool',(3,2,0)), 
            ('conv3', 'Conv', (192, 384, 3, 1,1), 'Relu'), ('conv4', 'Conv', (384, 256, 3, 1,1),'Relu'),
            ('conv5', 'Conv', (256, 256, 3, 1,1), 'Relu', 'Maxpool',(3,2,0)),
            ('fc1', 'FC', (256*6*6, 4096), 'Relu'), ('fc2', 'FC', (4096, 4096), 'Relu'),('fc3', 'FC', (4096, 1000), 'Relu')]

VGG16=[('conv1_1', 'Conv', (3, 64, 3, 1,0), 'Batchnorm','Relu'), ('conv1_2', 'Conv', (64, 64, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv2_1', 'Conv', (64, 128, 3, 1,0), 'Relu'), ('conv2_2', 'Conv', (128, 128, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv3_1', 'Conv', (128, 256, 3, 1,0), 'Relu'), ('conv3_2', 'Conv', (256, 256, 3, 1,1),'Relu'), ('conv3_3', 'Conv', (256, 256, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv4_1', 'Conv', (256, 512, 3, 1,0), 'Relu'), ('conv4_2', 'Conv', (512, 512, 3, 1,1),'Relu'), ('conv4_3', 'Conv', (512, 512, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv5_1', 'Conv', (512, 512, 3, 1,0), 'Relu'), ('conv5_2', 'Conv', (512, 512, 3, 1,1),'Relu'), ('conv5_3', 'Conv', (512, 512, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('fc1', 'FC', (512*7*7, 4096), 'Relu'), ('fc2', 'FC', (4096, 4096), 'Relu'),('fc3', 'FC', (4096, 1000))]


def train(args, config, tx2_model, device, tx2_train_loader, tx2_test_loader, tx2_optimizer, epoch):

    vm_start = time.time()
    tx2_model.train()
    train_loss = 0.0
    samples_num = 0
    for li_idx in range(args.local_iters):
        vm_data, vm_target = next(tx2_train_loader)
                
        vm_data, vm_target = vm_data.to(device), vm_target.to(device)
        vm_output = tx2_model(vm_data)

        tx2_optimizer.zero_grad()
        
        vm_loss = F.nll_loss(vm_output, vm_target)
        vm_loss.backward()
        tx2_optimizer.step()

        train_loss += (vm_loss.item() * vm_data.size(0))
        samples_num += vm_data.size(0)
        del vm_data
        del vm_target
        del vm_output
        del vm_loss

    if samples_num != 0:
        train_loss /= samples_num

    print('-->[{}] Train Epoch: {} train_loss: {:.6f}'.format(
                time_since(vm_start), epoch, train_loss))

def darts_train(args, config, model, device, train_loader, optimizer, epoch, lr):
    criterion = nn.CrossEntropyLoss().cuda()
    train_loss = 0.0
    train_samples_num = 0
    train_correct = 0
    time_start = time.time()
    model.train()

    for local_iter in range(args.local_iters):
        input, target = next(train_loader)
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += (loss.item() * input.size(0))
        train_samples_num += input.size(0)

        pred = output.argmax(1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        train_correct += correct

        del input
        del target
        del output
        del loss
    print('-->[{}] Train Epoch: {}  Train Loss: {:6f}'.format(time_since(time_start), epoch, train_loss))
    print('Train Accuracy [{}/{}] {:.2f}%'.format(train_correct, train_samples_num, np.float(1.0 * train_correct / train_samples_num) * 100.))

def darts_train_search(args, config, model, device, train_loader, valid_loader, optimizer, epoch, lr):
    architect = Architect(model, args)
    criterion = nn.CrossEntropyLoss().cuda()

    train_loss = 0.0
    train_samples_num = 0
    valid_samlpes_num = 0

    train_correct = 0
    valid_correct = 0
    
    time_start = time.time()
    model.train()

    for local_iter in range(args.local_iters):

        input, target = next(train_loader)
        input, target = input.to(device), target.to(device)

        input_search, target_search = next(valid_loader)
        input_search, target_search = input_search.to(device), target_search.to(device)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=False)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += (loss.item() * input.size(0))
        train_samples_num += input.size(0)
        valid_samlpes_num += input_search.size(0)

        pred = output.argmax(1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        train_correct += correct

        output_search = model(input_search)
        pred = output_search.argmax(1, keepdim=True)
        correct = pred.eq(target_search.view_as(pred)).sum().item()
        valid_correct += correct

        del input
        del target
        del input_search
        del target_search
        del output
        del output_search
        del loss
    
    print('-->[{}] Train Epoch: {}  Train Loss: {:6f}'.format(time_since(time_start), epoch, train_loss))
    print('Train Accuracy [{}/{}] {:.2f}%   Valid Accuracy [{}/{}] {:.2f}%'.format(train_correct, 
                                                                                   train_samples_num, 
                                                                                   np.float(1.0 * train_correct / train_samples_num) * 100.,
                                                                                   valid_correct,
                                                                                   valid_samlpes_num,
                                                                                   np.float(1.0 * valid_correct / valid_samlpes_num) * 100.))




def test(args, start, model, device, test_loader, epoch):
    model.eval()

    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    data = data.squeeze(1) 
                    data = data.view(-1, 28 * 28)
                else:
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    data = data.view(-1, 32, 32 * 3)                    
                else:
                    pass  

            if args.model_type == 'LSTM':
                hidden = model.initHidden(args.test_batch_size)
                hidden = hidden.send(data.location)
                for col_idx in range(32):
                    data_col = data[:, col_idx, :]
                    output, hidden = model(data_col, hidden)
            else:
                output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
                
            del data
            del target
            del output
            del pred
            del batch_correct

    test_loss /= len(test_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

    if args.enable_vm_test:  
        print('-->[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy))
    else:
        print('[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy))

    gc.collect()

    return test_loss, test_accuracy