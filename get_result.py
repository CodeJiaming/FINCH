import pandas as pd
import re

worker_num = 10

result_list = []
worker_result_list = [[] for i in range(worker_num)]

with open('./resluts.txt') as f:
  for line in f:
    if 'total bandwith consume' in line:
      bandwith_consume = re.findall(r'-?\d+\.?\d*', line)
    if 'Epoch' in line:
      data = re.findall(r'-?\d+\.?\d*', line)
      data.extend(bandwith_consume)
      result_list.append(data)
    if 'worker' in line:
      data = re.findall(r'-?\d+\.?\d*', line)
      worker_idx = int(data[0])
      data = data[1:]
      worker_result_list[worker_idx].append(data)

f.close()

for idx in range(len(result_list)):
  result_list[idx] = list(map(eval, result_list[idx]))


for worker_idx in range(worker_num):
  with open('./client_module/client_{}_log.txt'.format(worker_idx)) as f:
    epoch = 0
    for line in f:
      if 'Loss' in line:
        epoch += 1
        Loss = re.findall(r'-?\d+\.?\d*', line)[-1:]
      if 'Accuracy' in line:
        data = re.findall(r'-?\d+\.?\d*', line)
        data.extend(Loss)
        worker_result_list[worker_idx][epoch - 1].insert(0, str(epoch))
        worker_result_list[worker_idx][epoch - 1].extend(data)
      if 'error' in line:
        f.close()
        break

for worker_idx in range(worker_num):
  for idx in range(len(worker_result_list[worker_idx])):
    worker_result_list[worker_idx][idx] = list(map(eval, worker_result_list[worker_idx][idx]))
  
server_df = pd.DataFrame(result_list, columns=['epoch', 
                                              'Test Loss', 
                                              'test correct number', 
                                              'test number', 
                                              'accuracy(%)', 
                                              'bandwidth consume(MB)'], )


worker_dfs = list()
for worker_idx in range(worker_num):
  df = pd.DataFrame(worker_result_list[worker_idx], columns=['epoch', 
                                                             'train time(s)', 
                                                             'download time(s)', 
                                                             'upload time(s)', 
                                                             'train correct number', 
                                                             'train number', 
                                                             'train accuracy(%)', 
                                                             'valid correct number',
                                                             'valid number',
                                                             'valid accuracy(%)',
                                                             'train loss'])
  worker_dfs.append(df)


with pd.ExcelWriter('Result.xlsx') as writer:
  server_df.to_excel(writer, sheet_name='server', index=None)
  for worker_idx in range(worker_num):
    worker_dfs[worker_idx].to_excel(writer, sheet_name='worker{}'.format(worker_idx), index=None)

