import socket
import os
from queue import Queue
import argparse
import numpy as np
import copy
import re

import torch
import torch.nn.functional as F

from config import *
from communication_module.comm_utils import *
from training_module import datasets, utils
from training_module.action import ServerAction

from nas_module.darts.model_search import Network
from nas_module.darts.model import NetworkCIFAR as SinglePathNetwork


parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--alpha', type=float, default=200)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--mode', type=str, default='adaptive')
parser.add_argument('--topology', type=str, default='ring')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--local_updates', type=int, default=50)
parser.add_argument('--time_budget', type=float, default=50000)
parser.add_argument('--L', type=float, default=10)
parser.add_argument('--heter', type=str, default='medium')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SERVER_IP = "127.0.0.1"


# Compute the memory size of the model
def get_model_size(model):
    init_vector = torch.nn.utils.parameters_to_vector(model.parameters())
    model_size = init_vector.nelement() * 4 / 1024 / 1024
    return model_size


def main():
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    

    # init config
    common_config = CommonConfig()
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.epoch = args.epoch
    common_config.learn_rate = args.lr
    common_config.local_iters = args.local_updates

    device = torch.device("cuda" if common_config.use_cuda and torch.cuda.is_available() else "cpu")

    worker_idx_list = [0,1,2,3,4,5,6,7,8,9]
    client_port=[45001, 45002, 45003, 45004, 45005, 45006, 45007, 45008, 45009, 45010]
    master_port=[55001, 55002, 55003, 55004, 55005, 55006, 55007, 55008, 55009, 55010]


    for idx, worker_idx in enumerate(worker_idx_list):
        custom = dict()
        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       master_ip_addr=socket.gethostbyname(socket.gethostname()),
                                       action=ClientAction.LOCAL_TRAINING,
                                       custom=custom),
            ip_addr=WORKER_IP_LIST[idx],
            master_port=master_port[idx],
            client_port=client_port[idx],
            common_config=common_config)
        )
    worker_num = len(common_config.worker_list)

    # initialize a super-net as global model
    if args.dataset_type == 'CIFAR10':
        global_model = Network(layers=5)
    elif args.dataset_type == 'CIFAR100':
        global_model = Network(layers=8)
    global_model = global_model.to(device)

    masks = [torch.load('./subnet_mask/subnet-{}.pt'.format(idx)) for idx in range(worker_num)]
    vm_models = [copy.deepcopy(global_model) for idx in range(worker_num)]
    vm_masks = [copy.deepcopy(global_model.mask) for idx in range(worker_num)]

    clusters = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    pps_list = []

    print('There are {} clusters:'.format(len(clusters)))
    for cluster_idx, cluster in enumerate(clusters):

        print('----------------------')
        print('Cluster-{}: {}'.format(cluster_idx, cluster))
        print('Normal cell:')
        print(masks[cluster_idx][0])
        print('Reduce cell:')
        print(masks[cluster_idx][1])
        print('----------------------')
        pps_list.append(cluster[0])

        for worker_idx in cluster:
            vm_masks[worker_idx] = masks[cluster_idx]


    global_mask = vm_masks[0]
    for idx in range(1, worker_num):
        global_mask[0].data += vm_masks[idx][0].data
        global_mask[1].data += vm_masks[idx][1].data

    for k in range(len(global_mask)):
        for i in range(len(global_mask[k])):
            for j in range(len(global_mask[k][i])):
                if global_mask[k][i][j] > 0.:
                    global_mask[k][i][j] = 1.

    print('----------------------')
    print('global model: ')
    print('Normal cell:')
    print(global_mask[0])
    print('Reduce cell:')
    print(global_mask[1])
    print('----------------------')

    
    global_model.set_mask(global_mask)
    for idx in range(worker_num):
        vm_models[idx].set_mask(vm_masks[idx])

    
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, args.data_pattern)


    for worker_idx, worker in enumerate(common_config.worker_list):

            worker.config.mask = vm_masks[worker_idx]
            worker.config.para = vm_models[worker_idx].state_dict()
            worker.config.alpha = vm_models[worker_idx].arch_parameters()

            worker.config.custom["dataset_type"] = common_config.dataset_type
            worker.config.custom["batch_size"] = common_config.batch_size
            worker.config.custom["learn_rate"] = common_config.learn_rate
            worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
            worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

            worker.connection = connect_get_socket("127.0.0.1", worker.master_port)
            print(worker.config.idx, "Connection success!")


    # Create dataset instance
    test_loader = utils.create_dataloaders(test_dataset, batch_size=128, shuffle=False)
    print("test dataset:")
    utils.count_dataset(test_loader)

    global_para = global_model.state_dict()
    action_queue = Queue()


    for epoch_idx in range(1, 1 + common_config.epoch):

        
        # prepare the message to be sent to workers (woker.config.***)
        for idx, worker in enumerate(common_config.worker_list):

            vm_models[idx].load_state_dict(worker.config.para)
            vm_models[idx].set_mask(worker.config.mask)

            worker.config.para = global_model.state_dict()
            worker.config.alpha = global_model.arch_parameters()
            worker.config.mask = vm_masks[idx]
            worker.config.epoch_num = epoch_idx
        

        # send state to worker
        print("before send")
        action_queue.put(ServerAction.SEND_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)
        print("after send")

        # get state from worker
        print("before get")
        action_queue.put(ServerAction.GET_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)
        print("after get")

        
        # AGGREGATE
        with torch.no_grad():   

            # aggregate alpha
            update_alpha = copy.deepcopy(global_model.arch_parameters())
            total_data_num = copy.deepcopy(global_model.get_mask())
            
            for idx, worker in enumerate(common_config.worker_list):
                data_num = len(worker.config.custom["train_data_idxes"])
                alpha = worker.config.alpha
                mask = worker.config.mask

                if idx == 0:
                    for i in range(2):
                        update_alpha[i] = torch.mul(alpha[i], data_num)
                        total_data_num[i] = torch.mul(mask[i], data_num)
                else:
                    for i in range(2):
                        update_alpha[i] += torch.mul(alpha[i], data_num)
                        total_data_num[i] += torch.mul(mask[i], data_num)
                
                for i in range(2):
                    update_alpha[i] /= total_data_num[i]
                    update_alpha[i] = torch.where(torch.isnan(update_alpha[i]), torch.full_like(update_alpha[i], 0), update_alpha[i])
            
            global_model.set_alpha(update_alpha)
                
            

            # aggregate weight
            update_para = dict()
            global_para = global_model.state_dict()
            for name in global_para.keys():
                update_para[name] = [0., 0.]
            for idx, worker in enumerate(common_config.worker_list):
                data_num = len(worker.config.custom["train_data_idxes"])
                para = worker.config.para
                [mask_noraml, mask_reduce] = worker.config.mask
                for name, value in para.items():
                    if 'cell' not in name or 'preprocess' in name:
                        update_para[name][0] += value * data_num
                        update_para[name][1] += data_num
                    else:
                        cell_idx = int(re.findall(r'\d+', name)[0])
                        [i, j] = list(map(int, re.findall(r'\d+', name)[1: 3]))
                        if cell_idx == 2:
                            if mask_reduce[i][j] == 1.:
                                update_para[name][0] += value * data_num
                                update_para[name][1] += data_num
                        else:
                            if mask_noraml[i][j] == 1.:
                                update_para[name][0] += value * data_num
                                update_para[name][1] += data_num
                
            

            for name, value in update_para.items():
                assert name in global_para
                if value[1] > 0.:
                    update_para[name] = value[0] / value[1]
                elif value[1] == 0.:
                    update_para[name] = global_para[name]
            
            global_model.load_state_dict(update_para)
        

        # Test the global model
        global_model = global_model.to(device)
        global_model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()
                correct += batch_correct

        test_loss /= len(test_loader.dataset)
        test_accuracy = float(1.0 * correct / len(test_loader.dataset))


        comm_amount = []
        for idx, worker in enumerate(common_config.worker_list):
            print('worker{:}- trian time: {:}s, download time: {:}s, upload time: {:}s'.
                  format(idx, worker.config.train_time, worker.config.download_time, worker.config.upload_time) )
            if idx in pps_list:
                comm_amount.append(vm_models[idx].get_model_size())
        for idx, item in enumerate(comm_amount):
            print('subent{} size: {} MB'.format(idx, item))
        print('total bandwith consume: {} MB'.format(sum(comm_amount)))


        if epoch_idx == 1:
            test_loss_list = [test_loss]
            test_accuracy_list = [test_accuracy]
        else:
            if test_loss < min(test_loss_list) and test_accuracy > max(test_accuracy_list):
                print('---------------------------CheckPoint---------------------------------')
                print('Finding the best model at epoch {}, Test Loss={:.2f}, Accuracy: {}/{}({:.2f}%)'.format(
                    epoch_idx, test_loss, correct, len(test_loader.dataset), 100. * test_accuracy))
                print('model size: {}'.format(get_model_size(SinglePathNetwork(global_model.genotype()))))
                print('----------------------------------------------------------------------\n')

                for file_name in os.listdir('./checkpoint'):
                    os.remove(os.path.join('./checkpoint', file_name))
                
                suffix = '-Epoch{}Loss{:2f}Correct{}.pt'.format(epoch_idx, test_loss, correct)
                torch.save(global_model.state_dict(), './checkpoint/MODEL' + suffix)
                torch.save(global_model.alphas_normal, './checkpoint/ALPHA_NORMAL' + suffix)
                torch.save(global_model.alphas_reduce, './checkpoint/ALPHA_REDUCE' + suffix)
                torch.save(global_model.mask_normal, './checkpoint/MASK_NORMAL' + suffix)
                torch.save(global_model.mask_reduce, './checkpoint/MASK_REDUCE' + suffix)

            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)

    for worker in common_config.worker_list:
        worker.connection.shutdown(2)
        worker.connection.close()

def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)


    if dataset_type == "CIFAR100":
        # Every client missing "data_pattern * 10" class of data
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num-data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx*worker_num:(tmp_idx+1)*worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
    

    elif dataset_type == "CIFAR10":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)

        if data_pattern == 0:
            # IID
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)


        elif data_pattern == 2:
            # Every client missing 2 class of data
            partition_sizes = [ [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0  ],
                                [0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0  ],
                                [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125] ]
                                
        elif data_pattern == 4:
            # Every client missing 4 class of data
            partition_sizes = [ [0.166, 0.166, 0.166, 0.166, 0.166, 0.17,  0.0,   0.0,   0.0,   0.0  ],
                                [0.0,   0.166, 0.166, 0.166, 0.166, 0.166, 0.17,  0.0,   0.0,   0.0  ],
                                [0.0,   0.0,   0.166, 0.166, 0.166, 0.166, 0.166, 0.17,  0.0,   0.0  ],
                                [0.0,   0.0,   0.0,   0.166, 0.166, 0.166, 0.166, 0.166, 0.17,  0.0  ],
                                [0.0,   0.0,   0.0,   0.0,   0.166, 0.166, 0.166, 0.166, 0.166, 0.17 ],
                                [0.17,  0.0,   0.0,   0.0,   0.0,   0.166, 0.166, 0.166, 0.166, 0.166],
                                [0.166, 0.17,  0.0,   0.0,   0.0,   0.0,   0.166, 0.166, 0.166, 0.166],
                                [0.166, 0.166, 0.17,  0.0,   0.0,   0.0,   0.0,   0.166, 0.166, 0.166],
                                [0.166, 0.166, 0.166, 0.17,  0.0,   0.0,   0.0,   0.0,   0.166, 0.166],
                                [0.166, 0.166, 0.166, 0.166, 0.166, 0.17,  0.0,   0.0,   0.0,   0.0  ] ]
    
        elif data_pattern == 6:
            # Every client missing 6 class of data
            partition_sizes = [ [0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
                                [0.0,  0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0,  0.0 ],
                                [0.0,  0.0,  0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0 ],
                                [0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0 ],
                                [0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25, 0.0,  0.0 ],
                                [0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25, 0.0 ],
                                [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25],
                                [0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25],
                                [0.25, 0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.25],
                                [0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.25] ]

    for partition_size in partition_sizes:
        print(partition_size)
    
    train_data_partition = utils.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = utils.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
    
    return train_data_partition, test_data_partition

if __name__ == "__main__":
    main()
