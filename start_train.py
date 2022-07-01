import os
import time
import shutil
import subprocess

python_path = '...'
source_code_path = '...'

kill_server_cmd = """ pkill -f "server\.py --batch_size.*" """
kill_client_cmd = """ pkill -f "client\.py --master_ip.*" """


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# create exp dir and backup codes
def backup_codes(source_path, exp_path):
    print(exp_path)
    if os.path.exists(exp_path):
        print("exp dir exist!")
        return False
    create_dir(os.path.join(exp_path, 'checkpoint'))
    for root, dirs, files in os.walk(source_path):
        if '/.' in root:
            continue
        for fl in files:
            fl_type = os.path.splitext(fl)[-1]
            if fl_type == '.py' or fl_type == '.json' or fl_type == '.pt':
                dst_dir = root.replace(source_path, exp_path)
                create_dir(dst_dir)
                src_file = os.path.join(root, fl)
                dst_file = os.path.join(dst_dir, fl)
                shutil.copy(src_file, dst_file)
    return True


def excute_func(dataset_types, lrs, data_pattern, epoch, batch_size, local_iters):
    for bs in batch_size:
        for lr in lrs:
            for local_iter in local_iters:
                for dataset_type in dataset_types:
                    for dt in data_pattern:
                        complete = False
                        repeat = 0
                        while not complete:
                            os.system(kill_server_cmd)
                            os.system(kill_client_cmd)
                            exp_result_path = '../Result/'\
                                    'datatype{}_datapattern{}_lr{}_localsteps{}_batchsize{}'\
                                    .format(dataset_type, dt, lr, local_iter, bs)
                                
                            if backup_codes(source_code_path, exp_result_path):
                                cmd = 'cd ' + exp_result_path + ";" \
                                        + python_path + ' -u server.py --batch_size ' + str(bs) \
                                        +  ' --dataset_type ' + dataset_type + ' --lr ' + str(lr) \
                                        + ' --local_updates ' + str(local_iter) + ' --data_pattern ' + str(dt) \
                                        + ' --epoch ' + str(epoch) + ' > resluts.txt'
                                time_start = time.time()
                                result = subprocess.call(cmd, shell=True)
                                

                                if time.time() - time_start < 300:
                                    shutil.rmtree(exp_result_path)
                                else:
                                    complete = True
                            else:
                                complete = True
                            repeat += 1
                            if repeat > 3:
                                break


dataset_types = ["CIFAR10"]     # "CIFAR10", "CIFAR100"
lrs = [0.1]
data_pattern = [0]
epoch = 200
batch_size = [16]
local_iters = [100]             # 0, 2, 4, 6
excute_func(dataset_types, lrs, data_pattern, epoch, batch_size, local_iters)
