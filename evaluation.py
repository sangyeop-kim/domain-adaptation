from glob import glob
from models import *
from utils import *
import pickle


def soft_multitask_evaluation(PATH, encoding_kernel=None, attn_kernel=None, fold=None):
    if encoding_kernel is None or attn_kernel is None:
        with open(glob(PATH+'/*.txt')[0], 'r') as f:
            txt = f.read()

        encoding_kernel = int(txt.split('\n')[-2].split(' : ')[-1])
        attn_kernel = int(txt.split('\n')[-1].split(' : ')[-1])
    
    model = SoftMultiTaskNet(encoding_kernel, attn_kernel).cuda()


    metric1 = Metric()
    metric2 = Metric()

    if fold is None:
        folds = [i for i in glob(PATH+'/*') if 'fold' in i]
    else:
        folds = ['%s%s_fold' %(PATH, fold)]

    for pth in folds:
        model.load_state_dict(torch.load(pth + '/saved_weight.pt'))

        pkl = glob(pth + '/*.pkl')

        with open(pkl[0], 'rb') as f:
            tst1 = pickle.load(f)

        with open(pkl[1], 'rb') as f:
            tst2 = pickle.load(f)
        
        for k, data in enumerate(tst1['dataloader']):
            model.eval()
            data[0] = tst1['scaler'].transform(data[0].view(data[0].size()[0], -1))
            data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                dtype=torch.float)
            data = [d.cuda() for d in data]
            pred1, _ = model(data[0], data[0])   # one-hot
            target1 = data[1]
            metric1.measure_metric(target1, torch.exp(pred1))

        for k, data in enumerate(tst2['dataloader']):
            model.eval()  
            data[0] = tst2['scaler'].transform(data[0].view(data[0].size()[0], -1))
            data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                dtype=torch.float)
            data = [d.cuda() for d in data]
            _, pred2 = model(data[0], data[0])   # one-hot
            target2 = data[1]
            metric2.measure_metric(target2, torch.exp(pred2))

    metric1.save_metric()
    metric2.save_metric()

    acc1, prec1, recall1, F1, AUC1 = metric1.get_metric()
    print('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc1, prec1, recall1, F1, AUC1))
    acc2, prec2, recall2, F2, AUC2 = metric2.get_metric()
    print('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc2, prec2, recall2, F2, AUC2))

    return metric1, metric2


def Iterative_multitask_evaluation(PATH, encoding_kernel=None, attn_kernel=None, fold=None):
    if encoding_kernel is None or attn_kernel is None:
        with open(glob(PATH+'/*.txt')[0], 'r') as f:
            txt = f.read()

        encoding_kernel = int(txt.split('\n')[-2].split(' : ')[-1])
        attn_kernel = int(txt.split('\n')[-1].split(' : ')[-1])
    
    model = Iterative_MultiTaskNet(encoding_kernel, attn_kernel).cuda()


    metric1 = Metric()
    metric2 = Metric()
    
    if fold is None:
       folds = [i for i in glob(PATH+'/*') if 'fold' in i]
    else:
        folds = ['%s%s_fold' %(PATH, fold)]

    for pth in folds:
        model.load_state_dict(torch.load(pth + '/saved_weight.pt'))

        pkl = glob(pth + '/*.pkl')

        with open(pkl[0], 'rb') as f:
            tst1 = pickle.load(f)

        with open(pkl[1], 'rb') as f:
            tst2 = pickle.load(f)
        
        for k, data in enumerate(tst1['dataloader']):
            model.eval()
            data[0] = tst1['scaler'].transform(data[0].view(data[0].size()[0], -1))
            data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                dtype=torch.float)
            data = [d.cuda() for d in data]
            pred1 = model(data[0], 1)   # one-hot
            target1 = data[1]
            metric1.measure_metric(target1, torch.exp(pred1))

        for k, data in enumerate(tst2['dataloader']):
            model.eval()  
            data[0] = tst2['scaler'].transform(data[0].view(data[0].size()[0], -1))
            data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                dtype=torch.float)
            data = [d.cuda() for d in data]
            pred2 = model(data[0], 2)   # one-hot
            target2 = data[1]
            metric2.measure_metric(target2, torch.exp(pred2))

    metric1.save_metric()
    metric2.save_metric()

    acc1, prec1, recall1, F1, AUC1 = metric1.get_metric()
    print('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc1, prec1, recall1, F1, AUC1))
    acc2, prec2, recall2, F2, AUC2 = metric2.get_metric()
    print('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc2, prec2, recall2, F2, AUC2))

    return metric1, metric2



def single_evaluation(PATH, encoding_kernel=None, attn_kernel=None, fold=None):
    if encoding_kernel is None or attn_kernel is None:
        with open(glob(PATH+'/*.txt')[0], 'r') as f:
            txt = f.read()

        encoding_kernel = int(txt.split('\n')[-2].split(' : ')[-1])
        attn_kernel = int(txt.split('\n')[-1].split(' : ')[-1])
    
    model = SingleTaskNet(encoding_kernel, attn_kernel).cuda()


    metric = Metric()

    if fold is None:
        folds = [i for i in glob(PATH+'/*') if 'fold' in i]
    else:
        folds = ['%s%s_fold' %(PATH, fold)]

    for pth in folds:
        model.load_state_dict(torch.load(pth + '/saved_weight.pt'))

        pkl = glob(pth + '/*.pkl')

        with open(pkl[0], 'rb') as f:
            tst1 = pickle.load(f)

        
        for k, data in enumerate(tst1['dataloader']):
            model.eval()
            data[0] = tst1['scaler'].transform(data[0].view(data[0].size()[0], -1))
            data[0] = torch.tensor(data[0].reshape(data[0].shape[0],-1,65), 
                                dtype=torch.float)
            data = [d.cuda() for d in data]
            pred = model(data[0], 1)   # one-hot
            target = data[1]
            metric.measure_metric(target, torch.exp(pred))


    metric.save_metric()

    acc, prec, recall, F1, AUC = metric.get_metric()
    print('recipe : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc, prec, recall, F1, AUC))
    

    return metric



def hard_multitask_evaluation(PATH, encoding_kernel=None, attn_kernel=None, fold=None):
    if encoding_kernel is None or attn_kernel is None:
        with open(glob(PATH+'/*.txt')[0], 'r') as f:
            txt = f.read()

        encoding_kernel = int(txt.split('\n')[-2].split(' : ')[-1])
        attn_kernel = int(txt.split('\n')[-1].split(' : ')[-1])
    
    model = HardMultiTaskNet(encoding_kernel, attn_kernel).cuda()


    metric1 = Metric()
    metric2 = Metric()

    if fold is None:
        folds = [i for i in glob(PATH+'/*') if 'fold' in i]
    else:
        folds = ['%s%s_fold' %(PATH, fold)]

    for pth in folds:
        model.load_state_dict(torch.load(pth + '/saved_weight.pt'))

        pkl = glob(pth + '/*.pkl')

        with open(pkl[0], 'rb') as f:
            tst1 = pickle.load(f)

        
        for k, data in enumerate(tst1['dataloader']):
            model.eval()
            data = [d.cuda() for d in data]
            pred1, pred2 = model(data[0], 1)   # one-hot
            pred1 = pred1[data[-1] == 1]
            pred2 = pred2[data[-1] == 2]
            target1 = data[1][data[-1] == 1]
            target2 = data[1][data[-1] == 2]

            metric1.measure_metric(target1, torch.exp(pred1))
            metric2.measure_metric(target2, torch.exp(pred2))

    metric1.save_metric()
    metric2.save_metric()

    acc1, prec1, recall1, F1, AUC1 = metric1.get_metric()
    print('recipe1 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc1, prec1, recall1, F1, AUC1))
    acc2, prec2, recall2, F2, AUC2 = metric2.get_metric()
    print('recipe2 : Acc : %.2f, Prec : %.2f, Rec : %.2f, F1 : %.2f, AUC : %.4f'%
        (acc2, prec2, recall2, F2, AUC2))

    return metric1, metric2

