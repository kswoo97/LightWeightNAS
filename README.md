# Lightweight NAS
### Overview
This software is part of the ***mutable neural network project***.

In this software, we provide a lightweight training strategy for neural architecture search (NAS). 

Our training strategy enables a NAS algorithm to accurately find good neural architectures with a small number of performance-known neural architectures (i.e., neural architectures equipped with ground-truth performance).

Therefore, we 

To this end, we first train a NAS method to better understand the structure of neural architectures and then train the NAS method to accurately predict the performance of a given neural architecture.

### Data
Our software supports three NAS datasets.
Dataset details are provided in the following link:
https://drive.google.com/file/d/1SbxB-Ww_D9IzFciVhC5Qvsl7e6fKgJ6g/view?usp=drive_link

### Code guideline

One can refer to ```run.py``` python file to see how our NAS algorithm works.
