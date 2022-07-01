# FINCH
Source code of 《FINCH: Enhancing Fedearted Learning with Hierachical Neural Architecture Search》

## Environment
We perform the code in a server with 4 NVIDIA GTX 2080Ti, whose software environment is Ubuntu 18.04, CUDA v10.0, cuDNN v7.5.0.

## Running the code
Before running the code, you need to add the python path of your environment at the line 6 of "start_train.py" and the line 56 of "config.py", which indicate the Python interpreter required to start the process of server and client respectively. 

You can fix some essential experiment settings in the end of "start_train.py", such as dataset type, learning rate, batch size...

After setting, you can run the code via the command:
```
python start_train.py
```
