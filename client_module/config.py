from typing import List

class ClientAction:
    LOCAL_TRAINING = "local_training"

class ServerAction:
    LOCAL_TRAINING = "local_training"




class ClientConfig:
    def __init__(self,
                 idx: int,
                 master_ip_addr: str,
                 action: str
                 ):
        self.idx = idx
        self.master_ip_addr = master_ip_addr
        self.action = action
        self.epoch_num: int = 1
        self.model = list()

        self.para = dict()
        self.mask = list()
        self.alpha = list()
        
        self.resource = {"CPU": "1"}
        self.custom = dict()
        self.acc: float = 6
        self.loss: float = 1
        self.download_time: int = 0
        self.upload_time: int = 0
        self.train_time: int = 0