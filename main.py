from training import *
from glob import glob
import argparse
import warnings
import pickle
import torch
import os
warnings.filterwarnings('ignore')

def main():
    
    if not os.path.isdir('./best_model'):
        os.mkdir('./best_model')
        
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)


    X1 = data['X1']
    X2 = data['X2']
    y1 = data['y1']
    y2 = data['y2']

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--encoding_kernel', type=int, default=10)
    parser.add_argument('--attn_kernel', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_cv', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--loss_check', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--n_tol', type=int, default=5)
    parser.add_argument('--model', type=str, default='iterative', choices=['iterative','hard',
                                                                           'soft','single1', 
                                                                           'single2'])

    args = parser.parse_args()
    
    model = args.model
    torch.cuda.set_device(device=args.device)
    kfold = args.kfold
    random_state = args.random_state
    batch_size = args.batch_size
    encoding_kernel = args.encoding_kernel
    attn_kernel = args.attn_kernel
    n_epoch = args.n_epoch
    n_cv = args.n_cv
    learning_rate = args.learning_rate
    step_size = args.step_size
    gamma = args.gamma
    loss_check = args.loss_check
    num_workers = args.num_workers
    n_tol = args.n_tol
    
    if model == 'iterative':
        metric1, metric2 = Iterative_multitask_training(X1, y1, X2, y2, n_cv, kfold, random_state,
                                                        batch_size, num_workers, encoding_kernel,
                                                        attn_kernel, learning_rate, step_size,
                                                        gamma, n_epoch, loss_check, n_tol)
        
    if model == 'hard':
        metric1, metric2 = hard_multitask_training(X1, y1, X2, y2, n_cv, kfold, random_state, 
                                                   batch_size, num_workers, encoding_kernel,
                                                   attn_kernel, learning_rate, step_size, gamma, 
                                                   n_epoch, loss_check, n_tol)
        
    if model == 'soft':
        metric1, metric2 = soft_multitask_training(X1, y1, X2, y2, n_cv, kfold, random_state,
                                                   batch_size, num_workers, encoding_kernel, 
                                                   attn_kernel, learning_rate, step_size, gamma, 
                                                   n_epoch, loss_check, n_tol)
        
    if model == 'single1':
        metric = single_training(X1, y1, n_cv, kfold, random_state, batch_size, num_workers, 
                                 encoding_kernel, attn_kernel,learning_rate, step_size, gamma, 
                                 n_epoch, loss_check, n_tol, 1)
        
    if model == 'single2':  
        metric = single_training(X2, y2, n_cv, kfold, random_state, batch_size, num_workers, 
                                 encoding_kernel, attn_kernel,learning_rate, step_size, gamma, 
                                 n_epoch, loss_check, n_tol, 2)
    
    
if __name__=="__main__":
	main()