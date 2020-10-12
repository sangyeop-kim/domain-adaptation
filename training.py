from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from datetime import datetime
from evaluation import *
from models import *
from utils import *
import pickle
import time
import os



def Iterative_multitask_training(X1, y1, X2, y2, n_cv, kfold, random_state, batch_size, num_workers, 
                   encoding_kernel, attn_kernel, learning_rate, step_size, gamma, n_epoch,
                   loss_check, n_tol):
    
    
    for i in range(n_cv):
        
        print()
        print(i+1)
        
        PATH = './best_model/iterative_%s/' % datetime.now().strftime("%y%m%d_%H:%M:%S")
        os.mkdir(PATH)
        print(PATH)
        
        start_time = time.time()
        cv = StratifiedKFold(n_splits=kfold, random_state=random_state)
        recipe1_trn = [trn for trn, tst in cv.split(X1,y1)]
        recipe1_tst = [tst for trn, tst in cv.split(X1,y1)]
        recipe2_trn = [trn for trn, tst in cv.split(X2,y2)]
        recipe2_tst = [tst for trn, tst in cv.split(X2,y2)]

        
        for j in range(kfold):
            print('\nkfold', j+1)
            min_val_loss = np.inf
            tol = 0

            fold = j + 1
            fold_name = '%s_fold' % fold

            if not os.path.isdir(PATH + fold_name):
                os.mkdir(PATH + fold_name)
            
            recipe1_trn_index, recipe1_val_index = train_val_split(kfold, recipe1_trn[j], y1)
            recipe2_trn_index, recipe2_val_index = train_val_split(kfold, recipe2_trn[j], y2)

            len1 = recipe1_trn_index.shape[0]
            len2 = recipe2_trn_index.shape[0]
            max_len = len1 if len1 > len2 else len2
            recipe1_trn_index = sampling(max_len, recipe1_trn_index, random_state)
            recipe2_trn_index = sampling(max_len, recipe2_trn_index, random_state)
            
            
            X1_trn = X1[recipe1_trn_index]
            y1_trn = y1[recipe1_trn_index]
            X1_val = X1[recipe1_val_index]
            y1_val = y1[recipe1_val_index]
            X1_trn, y1_trn = resample_data(X1_trn, y1_trn, pos_ratio=0.1)
            
            scaler1 = StandardScaler()
            scaler1.fit(X1_trn)
            X1_trn = scaler1.transform(X1_trn)
            num1 = np.ones([y1_trn.size])
            
            X2_trn = X2[recipe2_trn_index]
            y2_trn = y2[recipe2_trn_index]
            X2_val = X2[recipe2_val_index]
            y2_val = y2[recipe2_val_index]
            X2_trn, y2_trn = resample_data(X2_trn, y2_trn, pos_ratio=0.1)
            
            scaler2 = StandardScaler()
            scaler2.fit(X2_trn)
            X2_trn = scaler2.transform(X2_trn)
            num2 = np.ones([y2_trn.size]) + 1
            
            dataset_trn1 = MyVarDataSet(X1_trn, y1_trn, num1)
            dataset_trn2 = MyVarDataSet(X2_trn, y2_trn, num2)

            dataset_val1 = MyVarDataSet(X1_val, y1_val, 1)
            dataset_val2 = MyVarDataSet(X2_val, y2_val, 2)
            
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

            dataloader_val1 = DataLoader(dataset=dataset_val1, 
                                        batch_size=len(dataset_val1), 
                                        shuffle=False, 
                                        num_workers=num_workers)
            
            dataloader_val2 = DataLoader(dataset=dataset_val2, 
                                        batch_size=len(dataset_val2), 
                                        shuffle=False, 
                                        num_workers=num_workers)
            
            dataloader_tst1 = DataLoader(dataset=dataset_tst1, 
                                        batch_size=len(dataset_tst1), 
                                        shuffle=False, 
                                        num_workers=num_workers)
            
            dataloader_tst2 = DataLoader(dataset=dataset_tst2, 
                                        batch_size=len(dataset_tst2), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            with open('%s%s/dataloader1.pkl' % (PATH, fold_name), 'wb') as f:
                pickle.dump({'dataloader' : dataloader_tst1, 'scaler' : scaler1}, f)

            with open('%s%s/dataloader2.pkl' % (PATH, fold_name), 'wb') as f:
                pickle.dump({'dataloader' : dataloader_tst2, 'scaler' : scaler2}, f)

            
            model = Iterative_MultiTaskNet(encoding_kernel, attn_kernel).cuda()

            
            optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
            optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate)

            scheduler1 = StepLR(optimizer1, step_size=step_size, gamma=gamma)
            scheduler2 = StepLR(optimizer2, step_size=step_size, gamma=gamma)

            model.train()
            for epoch in range(1, n_epoch+1):
                
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
                        total_loss1 += loss1
                        total_loss2 += loss2
                        
                        
                    except StopIteration:
                        break
                
                for k, data in enumerate(dataloader_val1):
                    model.eval()
                    data[0] = scaler1.transform(data[0].view(data[0].size()[0], -1))
                    data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                        dtype=torch.float)
                    data = [d.cuda() for d in data]
                    pred = model(data[0], 1)   # one-hot
                    val_loss1= F.nll_loss(pred, data[1], weight=cost_penalty)

                for k, data in enumerate(dataloader_val2):
                    model.eval()  
                    data[0] = scaler2.transform(data[0].view(data[0].size()[0], -1))
                    data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                        dtype=torch.float)
                    data = [d.cuda() for d in data]
                    pred = model(data[0], 1)   # one-hot
                    val_loss2= F.nll_loss(pred, data[1], weight=cost_penalty)

                mean_val_loss = (val_loss1 + val_loss2) / 2

                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                    model_name = 'saved_weight.pt'
                    torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)
                    tol = 0
                else:
                    tol += 1
                    if tol > n_tol:
                        print('early stopping %s epoch' % epoch)
                        print()
                        break

                if epoch % loss_check == 0:
                    loss1 = total_loss1.item() / len(dataloader_trn1)
                    loss2 = total_loss2.item() / len(dataloader_trn1)
                    
                    print('\tepoch : %s, recipe1_loss : %.6f, recipe2_loss : %.6f' %\
                        (epoch, loss1, loss2))
                    
                    print('\tvalidation recipe1_loss : %.6f, recipe2_loss : %.6f' %(val_loss1, 
                                                                                    val_loss2))
                    print('\tvalidation recipe_mean_loss : %.6f' % mean_val_loss)

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
                    print('\ttest recipe1_loss : %.6f, recipe2_loss : %.6f' %(loss1, loss2))
                    print()
                    
                
                scheduler1.step()
                scheduler2.step()
                
            # model_name = 'saved_weight.pt'
            # torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)

            print('%s fold performance' % fold)
            Iterative_multitask_evaluation(PATH, encoding_kernel, attn_kernel, fold)

        print()
        print()
        print('total performance')
        metric1, metric2 = Iterative_multitask_evaluation(PATH, encoding_kernel, attn_kernel)
        end_time = time.time()
        print('elapsed time : %.2f sec' % (end_time - start_time))

        acc1, prec1, recall1, F1, AUC1 = metric1.get_metric()
        acc2, prec2, recall2, F2, AUC2 = metric2.get_metric()

        
        with open(PATH + 'performance.txt', 'w') as f:
            f.write('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc1, prec1, recall1, F1, AUC1))
            f.write('\n')
            f.write('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc2, prec2, recall2, F2, AUC2))
            f.write('\n')
            f.write('iterative')
            f.write('\n')
            f.write('encoding_kernel : %s' % encoding_kernel)
            f.write('\n')
            f.write('attn_kernel : %s' % attn_kernel)
            f.write('\n')
            f.write(str(end_time - start_time))

    
    return metric1, metric2


def single_training(X, y, n_cv, kfold, random_state, batch_size, num_workers, 
                   encoding_kernel, attn_kernel, learning_rate, step_size, gamma, n_epoch,
                   loss_check, n_tol, recipe_num):
    
    for i in range(n_cv):
        print()
        print(i+1)
        
        PATH = './best_model/single%s_%s/' % (recipe_num,  
                                              datetime.now().strftime("%y%m%d_%H:%M:%S"))
        os.mkdir(PATH)
        print(PATH)
        
        start_time = time.time()
        cv = StratifiedKFold(n_splits=kfold, random_state=random_state)
        recipe_trn = [trn for trn, tst in cv.split(X,y)]
        recipe_tst = [tst for trn, tst in cv.split(X,y)]

        
        
        for j in range(kfold):
            print('\nkfold', j+1)
            min_val_loss = np.inf
            tol = 0

            fold = j + 1
            fold_name = '%s_fold' % fold

            if not os.path.isdir(PATH + fold_name):
                os.mkdir(PATH + fold_name)

            recipe_trn_index, recipe_val_index = train_val_split(kfold, recipe_trn[j], y)

            X_trn = X[recipe_trn_index]
            y_trn = y[recipe_trn_index]
            X_val = X[recipe_val_index]
            y_val = y[recipe_val_index]

            X_trn, y_trn = resample_data(X_trn, y_trn, pos_ratio=0.1)
            
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn = scaler.transform(X_trn)
            num = np.ones([y_trn.size])
            
            dataset_trn = MyVarDataSet(X_trn, y_trn, num)
            dataset_val = MyVarDataSet(X_val, y_val, 1)
            
            dataset_tst = MyVarDataSet(X[recipe_tst[j]], y[recipe_tst[j]], 1)
            
            dataloader_trn = DataLoader(dataset=dataset_trn, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)

            dataloader_val = DataLoader(dataset=dataset_val, 
                                        batch_size=len(dataset_val), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            dataloader_tst = DataLoader(dataset=dataset_tst, 
                                        batch_size=len(dataset_tst), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            
            with open('%s%s/dataloader.pkl' % (PATH, fold_name), 'wb') as f:
                pickle.dump({'dataloader' : dataloader_tst, 'scaler' : scaler}, f)

            model = SingleTaskNet(encoding_kernel, attn_kernel).cuda()

            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            
            model.train()
            for epoch in range(1, n_epoch+1):
                
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

                        total_loss += loss
                        
                        
                    except StopIteration:
                        break
                
                for k, data in enumerate(dataloader_val):
                    model.eval()
                    data[0] = scaler.transform(data[0].view(data[0].size()[0], -1))
                    data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                        dtype=torch.float)
                    data = [d.cuda() for d in data]
                    pred = model(data[0], 1)   # one-hot
                val_loss= F.nll_loss(pred, data[1], weight=cost_penalty)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model_name = 'saved_weight.pt'
                    torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)
                    tol = 0
                else:
                    tol += 1
                    if tol > n_tol:
                        print('early stopping %s epoch' % epoch)
                        print()
                        break

                if epoch % loss_check == 0:
                    loss = total_loss.item() / len(dataloader_trn)
                    print('\tepoch : %s, recipe_loss : %.6f' % (epoch, loss))
                    
                    ##### record validation loss #####
                    
                    print('\tvalidation recipe_loss : %.6f' %(val_loss))

                    for k, data in enumerate(dataloader_tst):
                        model.eval()
                        data[0] = scaler.transform(data[0].view(data[0].size()[0], -1))
                        data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                            dtype=torch.float)
                        data = [d.cuda() for d in data]
                        pred = model(data[0], 1)   # one-hot
                    loss= F.nll_loss(pred, data[1], weight=cost_penalty)
                    print('\ttest recipe_loss : %.6f' %(loss))
                    print()

                
                scheduler.step()

            
            for k, data in enumerate(dataloader_tst):
                model.eval()
                data[0] = scaler.transform(data[0].view(data[0].size()[0], -1))
                data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                    dtype=torch.float)
                data = [d.cuda() for d in data]
                pred = model(data[0], 1)   # one-hot         

            print('%s fold performance' % fold)
            single_evaluation(PATH, encoding_kernel, attn_kernel, fold)

        print()
        print()
        print('total performance')
        metric = single_evaluation(PATH, encoding_kernel, attn_kernel)
        acc, prec, recall, F1, AUC = metric.get_metric()

        end_time = time.time()
        print('elapsed time : %.2f sec' % (end_time - start_time))
        
        
        with open(PATH + 'performance.txt', 'w') as f:
            f.write('recipe : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc, prec, recall, F1, AUC))
            f.write('\n')
            f.write('single%s' % recipe_num)
            f.write('\n')
            f.write('encoding_kernel : %s' % encoding_kernel)
            f.write('\n')
            f.write('attn_kernel : %s' % attn_kernel)
            f.write('\n')
            f.write(str(end_time - start_time))
    
    
    return metric 




def hard_multitask_training(X1, y1, X2, y2, n_cv, kfold, random_state, batch_size, num_workers, 
                   encoding_kernel, attn_kernel, learning_rate, step_size, gamma, n_epoch,
                   loss_check, n_tol):

    for i in range(n_cv):
        print()
        print(i+1)

        PATH = './best_model/hard_%s/' % datetime.now().strftime("%y%m%d_%H:%M:%S")
        os.mkdir(PATH)

        print(PATH)

        start_time = time.time()
        cv = StratifiedKFold(n_splits=kfold, random_state=random_state)
        recipe1_trn = [trn for trn, tst in cv.split(X1,y1)]
        recipe1_tst = [tst for trn, tst in cv.split(X1,y1)]
        recipe2_trn = [trn for trn, tst in cv.split(X2,y2)]
        recipe2_tst = [tst for trn, tst in cv.split(X2,y2)]


        for j in range(kfold):
            print('\nkfold', j+1)
            min_val_loss = np.inf
            tol = 0

            fold = j + 1
            fold_name = '%s_fold' % fold

            if not os.path.isdir(PATH + fold_name):
                os.mkdir(PATH + fold_name)

            recipe1_trn_index, recipe1_val_index = train_val_split(kfold, recipe1_trn[j], y1)
            recipe2_trn_index, recipe2_val_index = train_val_split(kfold, recipe2_trn[j], y2)

            X1_trn = X1[recipe1_trn_index]
            y1_trn = y1[recipe1_trn_index]
            X2_trn = X2[recipe2_trn_index]
            y2_trn = y2[recipe2_trn_index]

            X1_val = X1[recipe1_val_index]
            y1_val = y1[recipe1_val_index]
            X2_val = X2[recipe2_val_index]
            y2_val = y2[recipe2_val_index]

            X1_tst = X1[recipe1_tst[j]]
            y1_tst = y1[recipe1_tst[j]]
            X2_tst = X2[recipe2_tst[j]]
            y2_tst = y2[recipe2_tst[j]]

            X1_trn, y1_trn = resample_data(X1_trn, y1_trn, pos_ratio=0.1)

            scaler1 = StandardScaler()
            scaler1.fit(X1_trn)
            X1_trn = scaler1.transform(X1_trn)
            X1_val = scaler1.transform(X1_val)
            X1_tst = scaler1.transform(X1_tst)

            X2_trn, y2_trn = resample_data(X2_trn, y2_trn, pos_ratio=0.1)

            scaler2 = StandardScaler()
            scaler2.fit(X2_trn)
            X2_trn = scaler2.transform(X2_trn)
            X2_val = scaler2.transform(X2_val)
            X2_tst = scaler2.transform(X2_tst)
            
            dataset_trn = MultiDataset(X1_trn, y1_trn, X2_trn, y2_trn)
            dataset_val = MultiDataset(X1_val, y1_val, X2_val, y2_val)
            dataset_tst = MultiDataset(X1_tst, y1_tst, X2_tst, y2_tst) 

            dataloader_trn = DataLoader(dataset=dataset_trn, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)

            dataloader_val = DataLoader(dataset=dataset_val, 
                                        batch_size=len(dataset_val), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            dataloader_tst = DataLoader(dataset=dataset_tst, 
                                        batch_size=len(dataset_tst), 
                                        shuffle=False, 
                                        num_workers=num_workers)


            with open('%s%s/dataloader.pkl' % (PATH, fold_name), 'wb') as f:
                pickle.dump({'dataloader' : dataloader_tst}, f)

            model = HardMultiTaskNet(encoding_kernel, attn_kernel).cuda()


            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

            model.train()
            for epoch in range(1, n_epoch+1):
                
                iter_ = iter(dataloader_trn)

                total_loss = 0.0
                for ii in range(len(dataloader_trn)):
                    # get the inputs

                    try :
                        optimizer.zero_grad()

                        data = next(iter_)
                        data = [d.cuda() for d in data]
                        pred1, pred2 = model(data[0], 1)
                        
                        cost_penalty= torch.FloatTensor([1,2]).cuda()
                        loss1 = F.nll_loss(pred1, data[1], weight=cost_penalty)
                        loss2 = F.nll_loss(pred2, data[1], weight=cost_penalty)
                        
                        
                        weight1 = torch.sum(data[-1] == 1).item() / len(data[-1])
                        weight2 = 1 - weight1
                        
                        loss = weight1 * loss1 + weight2 * loss2
                        
                        loss.backward()
                        optimizer.step()

                        total_loss += loss


                    except StopIteration:
                        break
                
                for k, data in enumerate(dataloader_val):
                    model.eval()
                    data = [d.cuda() for d in data]
                    pred1, pred2 = model(data[0], 1)   # one-hot
                
                
                loss1 = F.nll_loss(pred1, data[1], weight=cost_penalty)
                loss2 = F.nll_loss(pred2, data[1], weight=cost_penalty)

                weight1 = torch.sum(data[-1] == 1).item() / len(data[-1])
                weight2 = 1 - weight1

                val_loss = weight1 * loss1 + weight2 * loss2

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model_name = 'saved_weight.pt'
                    torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)
                    tol = 0
                else:
                    tol += 1
                    if tol > n_tol:
                        print('early stopping %s epoch' % epoch)
                        print()
                        break

                if epoch % loss_check == 0:
                    loss = total_loss.item() / len(dataloader_trn)
                    print('\tepoch : %s, recipe_loss : %.6f' % (epoch, loss))

                    ##### record validation loss #####
                    
                    print('\tvalidation recipe_loss : %.6f' %(val_loss))

                    for k, data in enumerate(dataloader_tst):
                        model.eval()
                        data = [d.cuda() for d in data]
                        pred1, pred2 = model(data[0], 1)   # one-hot
                    
                    
                    loss1 = F.nll_loss(pred1, data[1], weight=cost_penalty)
                    loss2 = F.nll_loss(pred2, data[1], weight=cost_penalty)

                    weight1 = torch.sum(data[-1] == 1).item() / len(data[-1])
                    weight2 = 1 - weight1

                    loss = weight1 * loss1 + weight2 * loss2
                    print('\ttest recipe_loss : %.6f' %(loss))
                    print()
               
                scheduler.step()

            print('%s fold performance' % fold)
            hard_multitask_evaluation(PATH, encoding_kernel, attn_kernel, fold)

        print()
        print()
        print('total performance')

        metric1, metric2 = hard_multitask_evaluation(PATH, encoding_kernel, attn_kernel)
        acc1, prec1, recall1, F1, AUC1 = metric1.get_metric()
        acc2, prec2, recall2, F2, AUC2 = metric2.get_metric()
        
        end_time = time.time()
        print('elapsed time : %.2f sec' % (end_time - start_time))

        with open(PATH + 'performance.txt', 'w') as f:
            f.write('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc1, prec1, recall1, F1, AUC1))
            f.write('\n')
            f.write('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc2, prec2, recall2, F2, AUC2))
            f.write('\n')
            f.write('hard')
            f.write('\n')
            f.write('encoding_kernel : %s' % encoding_kernel)
            f.write('\n')
            f.write('attn_kernel : %s' % attn_kernel)
            f.write('\n')
            f.write(str(end_time - start_time))

    return metric1, metric2


def soft_multitask_training(X1, y1, X2, y2, n_cv, kfold, random_state, batch_size, num_workers, 
                   encoding_kernel, attn_kernel, learning_rate, step_size, gamma, n_epoch,
                   loss_check, n_tol):
    
    for i in range(n_cv):

        print()
        print(i+1)

        PATH = './best_model/soft_%s/' % datetime.now().strftime("%y%m%d_%H:%M:%S")
        os.mkdir(PATH)

        print(PATH)

        start_time = time.time()
        cv = StratifiedKFold(n_splits=kfold, random_state=random_state)
        recipe1_trn = [trn for trn, tst in cv.split(X1,y1)]
        recipe1_tst = [tst for trn, tst in cv.split(X1,y1)]
        recipe2_trn = [trn for trn, tst in cv.split(X2,y2)]
        recipe2_tst = [tst for trn, tst in cv.split(X2,y2)]


        for j in range(kfold):
            print('\nkfold', j+1)
            min_val_loss = np.inf
            tol = 0

            fold = j + 1
            fold_name = '%s_fold' % fold

            if not os.path.isdir(PATH + fold_name):
                os.mkdir(PATH + fold_name)
            
            recipe1_trn_index, recipe1_val_index = train_val_split(kfold, recipe1_trn[j], y1)
            recipe2_trn_index, recipe2_val_index = train_val_split(kfold, recipe2_trn[j], y2)

            len1 = recipe1_trn_index.shape[0]
            len2 = recipe2_trn_index.shape[0]
            max_len = len1 if len1 > len2 else len2
            recipe1_trn_index = sampling(max_len, recipe1_trn_index, random_state)
            recipe2_trn_index = sampling(max_len, recipe2_trn_index, random_state)

            X1_trn = X1[recipe1_trn_index]
            y1_trn = y1[recipe1_trn_index]
            X2_trn = X2[recipe2_trn_index]
            y2_trn = y2[recipe2_trn_index]

            X1_val = X1[recipe1_val_index]
            y1_val = y1[recipe1_val_index]
            X2_val = X2[recipe2_val_index]
            y2_val = y2[recipe2_val_index]

            X1_tst = X1[recipe1_tst[j]]
            y1_tst = y1[recipe1_tst[j]]
            X2_tst = X2[recipe2_tst[j]]
            y2_tst = y2[recipe2_tst[j]]

            X1_trn, y1_trn = resample_data(X1_trn, y1_trn, pos_ratio=0.1)
            
            scaler1 = StandardScaler()
            scaler1.fit(X1_trn)
            X1_trn = scaler1.transform(X1_trn)
            num1 = np.ones([y1_trn.size])

            X2_trn, y2_trn = resample_data(X2_trn, y2_trn, pos_ratio=0.1)

            scaler2 = StandardScaler()
            scaler2.fit(X2_trn)
            X2_trn = scaler2.transform(X2_trn)
            num2 = np.ones([y2_trn.size]) + 1

            cut_len1 = len(X1_trn)
            cut_len2 = len(X2_trn)

            cut = cut_len2 if cut_len1 > cut_len2 else cut_len1

            X1_trn = X1_trn[:cut]
            y1_trn = y1_trn[:cut]
            X2_trn = X2_trn[:cut]
            y2_trn = y2_trn[:cut]
            
            dataset_trn1 = MyVarDataSet(X1_trn, y1_trn, num1)
            dataset_trn2 = MyVarDataSet(X2_trn, y2_trn, num2)

            dataset_val1 = MyVarDataSet(X1_val, y1_val, 1)
            dataset_val2 = MyVarDataSet(X2_val, y2_val, 2)

            dataset_tst1 = MyVarDataSet(X1_tst, y1_tst, 1)
            dataset_tst2 = MyVarDataSet(X2_tst, y2_tst, 2)

            dataloader_trn1 = DataLoader(dataset=dataset_trn1, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)

            dataloader_trn2 = DataLoader(dataset=dataset_trn2, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)

            dataloader_val1 = DataLoader(dataset=dataset_val1, 
                                        batch_size=len(dataset_val1), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            dataloader_val2 = DataLoader(dataset=dataset_val2, 
                                        batch_size=len(dataset_val2), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            dataloader_tst1 = DataLoader(dataset=dataset_tst1, 
                                        batch_size=len(dataset_tst1), 
                                        shuffle=False, 
                                        num_workers=num_workers)

            dataloader_tst2 = DataLoader(dataset=dataset_tst2, 
                                        batch_size=len(dataset_tst2), 
                                        shuffle=False, 
                                        num_workers=num_workers)


            with open('%s%s/dataloader1.pkl' % (PATH, fold_name), 'wb') as f:
                pickle.dump({'dataloader' : dataloader_tst1, 'scaler' : scaler1}, f)

            with open('%s%s/dataloader2.pkl' % (PATH, fold_name), 'wb') as f:
                pickle.dump({'dataloader' : dataloader_tst2, 'scaler' : scaler2}, f)

            model = SoftMultiTaskNet(encoding_kernel, attn_kernel).cuda()


            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            
            model.train()

            for epoch in range(1, n_epoch+1):
                
                iter1 = iter(dataloader_trn1)
                iter2 = iter(dataloader_trn2)
                
                total_loss = 0.0
                for ii in range(len(dataloader_trn1)):
                    # get the inputs
                    try :
                        optimizer.zero_grad()

                        data1 = next(iter1)
                        data1 = [d.cuda() for d in data1]
                        data2 = next(iter2)
                        data2 = [d.cuda() for d in data2]
                        pred1, pred2 = model(data1[0], data2[0])

                        cost_penalty= torch.FloatTensor([1,2]).cuda()
                        loss1 = F.nll_loss(pred1, data1[1], weight=cost_penalty)
                        loss2 = F.nll_loss(pred2, data2[1], weight=cost_penalty)

                        l1 = len(dataloader_trn1)
                        l2 = len(dataloader_trn2)
                        
                        loss = (loss1 * l1 + loss2 * l2) / (l1 + l2)

                        loss.backward()
                        optimizer.step()

                        total_loss += loss


                    except StopIteration:
                        break
                
                for k, data in enumerate(dataloader_val1):
                    model.eval()
                    data[0] = scaler1.transform(data[0].view(data[0].size()[0], -1))
                    data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                        dtype=torch.float)
                    data = [d.cuda() for d in data]
                    pred, _ = model(data[0], data[0])   # one-hot
                    loss1= F.nll_loss(pred, data[1], weight=cost_penalty)

                for k, data in enumerate(dataloader_val2):
                    model.eval()  
                    data[0] = scaler2.transform(data[0].view(data[0].size()[0], -1))
                    data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                        dtype=torch.float)
                    data = [d.cuda() for d in data]
                    _, pred = model(data[0], data[0])   # one-hot
                    loss2= F.nll_loss(pred, data[1], weight=cost_penalty)
                    
                l1 = len(dataloader_val1)
                l2 = len(dataloader_val2)
                val_loss = (loss1 *  l1 + loss2 * l2) / (l1 + l2)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model_name = 'saved_weight.pt'
                    torch.save(model.state_dict(), PATH + fold_name + '/' + model_name)
                    tol = 0
                else:
                    tol += 1
                    if tol > n_tol:
                        print('early stopping %s epoch' % epoch)
                        print()
                        break

                if epoch % loss_check == 0:
                    loss = total_loss.item() / len(dataloader_trn1)
                    print('\tepoch : %s, recipe_loss : %.6f' % (epoch, loss))

                    ##### record validation loss #####
                    
                    print('\tvalidation recipe_loss : %.6f' %(val_loss))


                    for k, data in enumerate(dataloader_tst1):
                        model.eval()
                        data[0] = scaler1.transform(data[0].view(data[0].size()[0], -1))
                        data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                            dtype=torch.float)
                        data = [d.cuda() for d in data]
                        pred, _ = model(data[0], data[0])   # one-hot
                        loss1= F.nll_loss(pred, data[1], weight=cost_penalty)

                    for k, data in enumerate(dataloader_tst2):
                        model.eval()  
                        data[0] = scaler2.transform(data[0].view(data[0].size()[0], -1))
                        data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                            dtype=torch.float)
                        data = [d.cuda() for d in data]
                        _, pred = model(data[0], data[0])   # one-hot
                        loss2= F.nll_loss(pred, data[1], weight=cost_penalty)
                        
                    l1 = len(dataloader_tst1)
                    l2 = len(dataloader_tst2)
                    loss = (loss1 *  l1 + loss2 * l2) / (l1 + l2)
                    print('\ttest recipe_loss : %.6f' %(loss))
                    print()

                
                scheduler.step()


            print('%s fold performance' % fold)
            soft_multitask_evaluation(PATH, encoding_kernel, attn_kernel, fold)

        print()
        print()
        print('total performance')
        metric1, metric2 = soft_multitask_evaluation(PATH, encoding_kernel, attn_kernel)
        acc1, prec1, recall1, F1, AUC1 = metric1.get_metric()
        acc2, prec2, recall2, F2, AUC2 = metric2.get_metric()

        end_time = time.time()
        print('elapsed time : %.2f sec' % (end_time - start_time))
        
        with open(PATH + 'performance.txt', 'w') as f:
            f.write('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc1, prec1, recall1, F1, AUC1))
            f.write('\n')
            f.write('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
            (acc2, prec2, recall2, F2, AUC2))
            f.write('\n')
            f.write('soft')
            f.write('\n')
            f.write('encoding_kernel : %s' % encoding_kernel)
            f.write('\n')
            f.write('attn_kernel : %s' % attn_kernel)
            f.write('\n')
            f.write(str(end_time - start_time))

    return metric1, metric2