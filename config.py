import time
from typing import List
# from torch.utils.tensorboard import SummaryWriter
from communication_module.comm_utils import *

WORKER_IP_LIST = [
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
    "127.0.0.1",
]

class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    ip_addr = ""
    listen_port = 0
    master_port = 0

    __local_script_path = "./client.py"

    def __init__(self,
                 config,
                 ip_addr,
                 master_port,
                 client_port,
                 common_config
                 ):
        self.config = config
        self.work_thread = None
        self.idx = config.idx
        self.ip_addr = ip_addr
        self.client_port = client_port
        self.master_port = master_port
        self.connection = None
        self.common_config = common_config
        self.__start_local_worker_process()


    def __start_local_worker_process(self):
        python_path = '/data/yxu/software/Anaconda/envs/pytorch/bin/python'

        os.system('cd ' + os.getcwd() + '/client_module' + ';nohup  ' + python_path + ' -u client.py --master_ip ' 
                   + '127.0.0.1' + ' --master_port ' + str(self.master_port)  + ' --client_port ' + str(self.client_port)  + ' --idx ' + str(self.idx) 
                   + ' --dataset_type ' + str(self.common_config.dataset_type) + ' --batch_size ' + str(self.common_config.batch_size) 
                   + ' --local_iters ' + str(self.common_config.local_iters) + ' --lr ' + str(self.common_config.learn_rate) 
                   + ' > client_' + str(self.idx) + '_log.txt 2>&1 &')



    async def send_config(self):
        self.config.download_time = time.time()
        send_data_socket(self.config, self.connection)

    async def get_config(self):
        self.config = get_data_socket(self.connection)
        self.config.upload_time = time.time() - self.config.upload_time

    async def local_training(self):
        self.config.action = ClientAction.LOCAL_TRAINING
        print("before send", self.idx, self.ip_addr, self.listen_port, self.master_port)
        await self.send_config()
        print("after send", self.idx)
        recv_config = await self.get_config()
        print("after get", self.idx)
        self.config = recv_config

class CommonConfig:
    def __init__(self):

        self.dataset_type = 'CIFAR10'
        self.batch_size = 16
        self.learn_rate = 0.1
        self.use_cuda = True
        self.training_mode = 'local'
        self.local_iters = 1

        self.worker_num = len(WORKER_IP_LIST)

        self.epoch_start = 0
        self.epoch = 1000

        self.test_batch_size = 64

        self.master_listen_port_base = 57000
        self.client_listen_port_base = 47000

        self.available_gpu: list = ['0']

        self.worker_list: List[Worker] = list()



class ClientConfig:
    def __init__(self,
                 idx: int,
                 master_ip_addr: str,
                 action: str,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.master_ip_addr = master_ip_addr
        self.action = action
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()

        self.para = dict()
        self.mask = list()
        self.alpha = list()
        
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.download_time: int = 0
        self.upload_time: int = 0
        self.train_time: int = 0
