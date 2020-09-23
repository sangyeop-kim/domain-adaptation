from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from datetime import datetime
from models import *
from utils import *
import time
import os



def multi_training(X1, y1, X2, y2, n_cv, kfold, random_state, batch_size, num_workers, 
                   encoding_kernel, attn_kernel, learning_rate, step_size, gamma, n_epoch,
                   loss_check):
    
    
    for i in range(n_cv):
        
        print()
        print(i+1)
        
        PATH = './best_model/%s/' % datetime.now().strftime("%y%m%d_%H:%M:%S")
        os.mkdir(PATH)
        
        start_time = time.time()
        cv = StratifiedKFold(n_splits=kfold, random_state=random_state)
        recipe1_trn = [trn for trn, tst in cv.split(X1,y1)]
        recipe1_tst = [tst for trn, tst in cv.split(X1,y1)]
        recipe2_trn = [trn for trn, tst in cv.split(X2,y2)]
        recipe2_tst = [tst for trn, tst in cv.split(X2,y2)]
        
        metric1 = Metric()
        metric2 = Metric()
        
        for j in range(kfold):
            print('\nkfold', j+1)
            
            len1 = recipe1_trn[j].shape[0]
            len2 = recipe2_trn[j].shape[0]
            max_len = len1 if len1 > len2 else len2
            recipe1_trn[j] = sampling(max_len, recipe1_trn[j], random_state)
            recipe2_trn[j] = sampling(max_len, recipe2_trn[j], random_state)
            
            
            X1_trn = X1[recipe1_trn[j]]
            y1_trn = y1[recipe1_trn[j]]
            X1_trn, y1_trn = resample_data(X1_trn, y1_trn, pos_ratio=0.1)
            
            scaler1 = StandardScaler()
            scaler1.fit(X1_trn)
            X1_trn = scaler1.transform(X1_trn)
            num1 = np.ones([y1_trn.size])
            
            X2_trn = X2[recipe2_trn[j]]
            y2_trn = y2[recipe2_trn[j]]
            X2_trn, y2_trn = resample_data(X2_trn, y2_trn, pos_ratio=0.1)
            
            scaler2 = StandardScaler()
            scaler2.fit(X2_trn)
            X2_trn = scaler2.transform(X2_trn)
            num2 = np.ones([y2_trn.size]) + 1
            
            dataset_trn1 = MyVarDataSet(X1_trn, y1_trn, num1)
            dataset_trn2 = MyVarDataSet(X2_trn, y2_trn, num2)
            
            dataset_tst1 = MyVarDataSet(X1[recipe1_tst[j]], y1[recipe1_tst[j]], 1)
            dataset_tst2 = MyVarDataSet(X2[recipe2_tst[j]], y2[recipe2_tst[j]], 2)
            
            dataloader_trn1 = DataLoader(dataset=dataset_trn1, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)
            
            dataloader_trn2 = DataLoader(dataset=dataset_trn2, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)
            
            dataloader_tst1 = DataLoader(dataset=dataset_tst1, 
                                        batch_size=len(dataset_tst1), 
                                        shuffle=False, 
                                        num_workers=num_workers)
            
            dataloader_tst2 = DataLoader(dataset=dataset_tst2, 
                                        batch_size=len(dataset_tst2), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            
            model = Iterative_MultiTaskNet(encoding_kernel, attn_kernel).cuda()

            
            
            optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
            optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate)

            scheduler1 = StepLR(optimizer1, step_size=step_size, gamma=gamma)
            scheduler2 = StepLR(optimizer2, step_size=step_size, gamma=gamma)
            for epoch in range(1, n_epoch+1):
                metric1.initialize()
                metric2.initialize()
                iter1 = iter(dataloader_trn1)
                iter2 = iter(dataloader_trn2)
                
                total_loss1 = 0.0
                total_loss2 = 0.0
                for ii in range(len(dataloader_trn1)):
                    # get the inputs
                    
                    try :
                        optimizer1.zero_grad()

                        data1 = next(iter1)
                        data1 = [d.cuda() for d in data1]
                        pred1 = model(data1[0], 1)
                        cost_penalty= torch.FloatTensor([1,2]).cuda()
                        loss1 = F.nll_loss(pred1, data1[1], weight=cost_penalty)
                        loss1.backward()
                        optimizer1.step()

                        optimizer2.zero_grad()

                        data2 = next(iter2)
                        data2 = [d.cuda() for d in data2]
                        pred2 = model(data2[0], 2)
                        loss2= F.nll_loss(pred2, data2[1], weight=cost_penalty)
                        loss2.backward()
                        optimizer2.step()
                        
                        pred = torch.cat([pred1,pred2])
                        data_ = torch.cat([data1[1], data2[1]])
                        total_loss1 += loss1 * len(data1)
                        total_loss2 += loss2 * len(data2)
                        
                        
                    except StopIteration:
                        break
                    
                if epoch % loss_check == 0:
                    loss1 = total_loss1.item() / len(dataloader_trn1)
                    loss2 = total_loss2.item() / len(dataloader_trn1)
                    
                    print('epoch : %s, recipe1_loss : %.6f, recipe2_loss : %.6f' %\
                        (epoch, loss1, loss2))
                    
                    
                    for k, data in enumerate(dataloader_tst1):
                        model.eval()
                        data[0] = scaler1.transform(data[0].view(data[0].size()[0], -1))
                        data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                            dtype=torch.float)
                        data = [d.cuda() for d in data]
                        pred = model(data[0], 1)   # one-hot
                        loss1= F.nll_loss(pred, data[1], weight=cost_penalty)

                    for k, data in enumerate(dataloader_tst2):
                        model.eval()  
                        data[0] = scaler2.transform(data[0].view(data[0].size()[0], -1))
                        data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                            dtype=torch.float)
                        data = [d.cuda() for d in data]
                        pred = model(data[0], 1)   # one-hot
                        loss2= F.nll_loss(pred, data[1], weight=cost_penalty)
                    print('validation recipe1_loss : %.6f, recipe2_loss : %.6f' %(loss1, loss2))
                    print()
                    
                model.train()
                scheduler1.step()
                scheduler2.step()
                

            
            for k, data in enumerate(dataloader_tst1):
                model.eval()
                data[0] = scaler1.transform(data[0].view(data[0].size()[0], -1))
                data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                    dtype=torch.float)
                data = [d.cuda() for d in data]
                pred = model(data[0], 1)   # one-hot

                metric1.measure_metric(data[1], torch.exp(pred))

            for k, data in enumerate(dataloader_tst2):
                model.eval()  
                data[0] = scaler2.transform(data[0].view(data[0].size()[0], -1))
                data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                    dtype=torch.float)
                data = [d.cuda() for d in data]
                pred = model(data[0], 1)   # one-hot

                metric2.measure_metric(data[1], pred)
            
            metric1.save_metric()
            metric2.save_metric()
            
            fold = j + 1
            fold_name = '%s_fold' % fold
            
            if not os.path.isdir(PATH + fold_name):
                os.mkdir(PATH + fold_name)
                
            model_name = 'saved_weight.pt'
            torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)
            print('save_file:%s' % (PATH + fold_name + '/' + model_name))
        
        print()
        print()
        acc, prec1, recall1, F1, AUC = metric1.get_metric()
        print('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec1, recall1, F1, AUC))
        acc, prec2, recall2, F1, AUC = metric2.get_metric()
        print('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec2, recall2, F1, AUC))
        
        with open(PATH + 'performance.txt', 'w') as f:
            f.write('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec1, recall1, F1, AUC))
            f.write('\n')
            f.write('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec2, recall2, F1, AUC))
    
    
    return model, metric1, metric2


def single_training(X, y, n_cv, kfold, random_state, batch_size, num_workers, 
                   encoding_kernel, attn_kernel, learning_rate, step_size, gamma, n_epoch,
                   loss_check):
    
    for i in range(n_cv):
        print()
        print(i+1)
        
        PATH = './best_model/%s/' % datetime.now().strftime("%y%m%d_%H:%M:%S")
        os.mkdir(PATH)
        
        start_time = time.time()
        cv = StratifiedKFold(n_splits=kfold, random_state=random_state)
        recipe_trn = [trn for trn, tst in cv.split(X,y)]
        recipe_tst = [tst for trn, tst in cv.split(X,y)]

        
        metric = Metric()
        
        for j in range(kfold):
            print('\nkfold', j+1)
            
            
            X_trn = X[recipe_trn[j]]
            y_trn = y[recipe_trn[j]]
            X_trn, y_trn = resample_data(X_trn, y_trn, pos_ratio=0.1)
            
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn = scaler.transform(X_trn)
            num = np.ones([y_trn.size])
            
            dataset_trn = MyVarDataSet(X_trn, y_trn, num)
            
            dataset_tst = MyVarDataSet(X[recipe_tst[j]], y[recipe_tst[j]], 1)
            
            dataloader_trn = DataLoader(dataset=dataset_trn, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)

            
            dataloader_tst = DataLoader(dataset=dataset_tst, 
                                        batch_size=len(dataset_tst), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            

            model = SingleTaskNet(encoding_kernel, attn_kernel).cuda()

            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            
            for epoch in range(1, n_epoch+1):
                metric.initialize()
                iter_ = iter(dataloader_trn)
                
                total_loss = 0.0
                for ii in range(len(dataloader_trn)):
                    # get the inputs
                    
                    try :
                        optimizer.zero_grad()

                        data = next(iter_)
                        data = [d.cuda() for d in data]
                        pred = model(data[0], 1)
                        cost_penalty= torch.FloatTensor([1,2]).cuda()
                        loss = F.nll_loss(pred, data[1], weight=cost_penalty)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss * len(data)
                        
                        
                    except StopIteration:
                        break
                    
                if epoch % loss_check == 0:
                    loss = total_loss.item() / len(dataloader_trn)
                    print('epoch : %s, recipe_loss : %.6f' % (epoch, loss))
                    
                    ##### record validation loss #####
                    for k, data in enumerate(dataloader_tst):
                        model.eval()
                        data[0] = scaler.transform(data[0].view(data[0].size()[0], -1))
                        data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                            dtype=torch.float)
                        data = [d.cuda() for d in data]
                        pred = model(data[0], 1)   # one-hot
                    loss= F.nll_loss(pred, data[1], weight=cost_penalty)
                    print('validation recipe_loss : %.6f' %(loss))
                    print()

                model.train()
                scheduler.step()
                

            
            for k, data in enumerate(dataloader_tst):
                model.eval()
                data[0] = scaler.transform(data[0].view(data[0].size()[0], -1))
                data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                    dtype=torch.float)
                data = [d.cuda() for d in data]
                pred = model(data[0], 1)   # one-hot

                metric.measure_metric(data[1], torch.exp(pred))

            
            metric.save_metric()
            
            fold = j + 1
            fold_name = '%s_fold' % fold
            
            if not os.path.isdir(PATH + fold_name):
                os.mkdir(PATH + fold_name)
                
            model_name = 'saved_weight.pt'
            torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)
            print('save_file:%s' % (PATH + fold_name + '/' + model_name))

        
        print()
        print()
        acc, prec, recall, F1, AUC = metric.get_metric()
        print('recipe : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec, recall, F1, AUC))
        
        
        with open(PATH + 'performance.txt', 'w') as f:
            f.write('recipe : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec, recall, F1, AUC))
    
    
    return model, metric 