import random
import sys
import time
import argparse
import asyncio
import numpy as np

import torch
import torch.optim as optim


from config import ClientConfig
from client_comm_utils import *
from training_utils import darts_train, darts_train_search
import datasets
import utils

sys.path.append('..')
from nas_module.darts.model_search import Network
from nas_module.darts.architect import Architect

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="172.16.50.9",
                    help='IP address for controller or ps')
parser.add_argument('--client_port', type=int, default=47000, metavar='N',
                    help='Port used to listen msg from master')
parser.add_argument('--master_port', type=int, default=57000, metavar='N',
                    help='')


parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--min_lr', type=float, default=0.0001)
parser.add_argument('--decay_rate', type=float, default=1)
parser.add_argument('--local_iters', type=int, default=50)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--enable_vm_test', action="store_true", default=True)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--pattern_idx', type=int, default=0)
parser.add_argument('--tx_num', type=int, default=1)
parser.add_argument('--visible_cuda', type=str, default='-1')



parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--train_portion', type=float, default=0.75, help='portion of training data')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

args = parser.parse_args()


MASTER_IP = args.master_ip
LISTEN_PORT = args.client_port
MASTER_LISTEN_PORT = args.master_port

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(((int(args.idx)+3) % 4))
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    client_config = ClientConfig(
        idx=args.idx,
        master_ip_addr=args.master_ip,
        action=""
    )
    print("start")
    print(MASTER_IP, MASTER_LISTEN_PORT)
    connection = connect_send_socket(MASTER_IP, MASTER_LISTEN_PORT)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    task = asyncio.ensure_future(get_init_config(client_config, connection))
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    

    # data preparation
    train_dataset, test_dataset = datasets.load_datasets(client_config.custom["dataset_type"])

    train_data_idxes =  client_config.custom["train_data_idxes"]
    random.shuffle(train_data_idxes)
    num_train = len(train_data_idxes)
    split = int(np.floor(args.train_portion * num_train))
    valid_data_idxes = train_data_idxes[split:num_train]
    train_data_idxes = train_data_idxes[:split]
    
    train_loader = utils.create_dataloaders(train_dataset, 
                                            batch_size=client_config.custom["batch_size"], 
                                            selected_idxs=train_data_idxes)
    
    
    valid_loader = utils.create_dataloaders(train_dataset, 
                                           batch_size=client_config.custom["batch_size"], 
                                           selected_idxs=valid_data_idxes)
    
    test_loader = utils.create_dataloaders(test_dataset, 
                                           batch_size=client_config.custom["batch_size"], 
                                           selected_idxs=client_config.custom["test_data_idxes"], 
                                           shuffle=False)

    print("train dataset:")
    utils.count_dataset(train_loader)
    print("valid dataset:")
    utils.count_dataset(valid_loader)
    print("test dataset:")
    utils.count_dataset(test_loader)

    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(client_config, connection, train_loader, valid_loader, test_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()


async def local_training(config, conn, train_loader, valid_loader, test_loader):
    # model = MyNet(config.model)
    model = Network()
    model.load_state_dict(config.para)
    model.set_alpha(config.alpha)
    model.set_mask(config.mask)
    model = model.to(device)


    vm_lr = np.max((args.decay_rate ** (config.epoch_num - 1) * config.custom["learn_rate"], args.min_lr))
    optimizer = optim.SGD(
        model.parameters(),
        lr=vm_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)


    start_time = time.time()
    darts_train_search(args, config, model, device, train_loader, valid_loader, optimizer, config.epoch_num, vm_lr)
    config.train_time = time.time() - start_time
    
    config.alpha = model.arch_parameters()
    config.para = model.state_dict()
    config.mask = model.get_mask()

    print("before send")
    config.upload_time = time.time()
    send_data_socket(config, conn)
    print("after send")
    
    config_received = get_data_socket(conn)
    config_received.download_time = time.time() - config_received.download_time
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

async def get_init_config(config, conn):
    print("before init")
    print(LISTEN_PORT, MASTER_IP)
    config_received = get_data_socket(conn)
    config_received.download_time = time.time() - config_received.download_time
    print("after init")
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

if __name__ == '__main__':
    main()


