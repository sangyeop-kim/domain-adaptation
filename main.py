from training import *
import warnings
warnings.filterwarnings('ignore')
import pickle
import torch

torch.cuda.set_device(device=0)

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    
    
X1 = data['X1']
X2 = data['X2']
y1 = data['y1']
y2 = data['y2']


kfold = 3
random_state = 0
batch_size = 64
encoding_kernel = 10
attn_kernel = 10
n_epoch = 100 #100 
# verbose = True
n_cv = 10 #20
learning_rate = 0.0001
step_size = 30
gamma = 0.8
loss_check = 10
num_workers = 16
model_mode = 'multi'

if model_mode == 'single1':
    model1, metric1 = single_training(X1, y1, n_cv, kfold, random_state, batch_size, num_workers, 
                                encoding_kernel, attn_kernel, learning_rate,step_size, gamma, 
                                n_epoch, loss_check)

elif model_mode == 'single2':
    model2, metric2 = single_training(X2, y2, n_cv, kfold, random_state, batch_size, num_workers, 
                                encoding_kernel, attn_kernel, learning_rate,step_size, gamma, 
                                n_epoch, loss_check)
    
else:
    multi_model, multi_metric1, multi_metric2 = multi_training(X1, y1, X2, y2, n_cv, kfold, 
                                                           random_state, batch_size, num_workers, 
                                                           encoding_kernel, attn_kernel, 
                                                           learning_rate, step_size, gamma, 
                                                           n_epoch, loss_check)