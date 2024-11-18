# Lightweight NAS
### A lightweight training strategy for NAS. 

In this software, we provide a lightweight training strategy for neural architecture search (NAS). 

Specifically, we first train a NAS method to better understand the structure of neural architectures and then train the NAS method to accurately predict the performance of a given neural architecture.


## How to run our training method?

Neural architecture graphs should be located in ```./dataset``` folder.

### Running main.py

One can find the performance prediction results with this code:
```
python3 main.py -dname nb101 -device cuda:0 -train_ratio 0.01 -proj_layer 2 -proj_dim1 128 -proj_dim2 64 -ssl_lr 0.001 -wdecay 1e-6 -lamda1 0.5 -lamda2 0.5 -enc gatedgcn
```
