import numpy as np
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


def load_X(recipe_no, step_no):
    X_name = './rawdata/RECIPE{}_STEP{:02}.csv'.format(recipe_no, step_no)
    return np.genfromtxt(X_name, delimiter=',', dtype=np.float32)


def load_y(recipe_no):
    y_name = './rawdata/RECIPE{}_fault.csv'.format(recipe_no)
    return np.genfromtxt(y_name, delimiter=',', dtype=np.int32)



def load_data_cat_nan(recipe_no, step_no_list=[10, 11, 12, 13, 14], 
                      preprocessing=None, scale=True, split=True,
                      trn_size=0.6, val_size=0.2, random_state=0):
    # fault -> array
    y = load_y(recipe_no)
    X_list = []
    # step 10~14까지 가져오기
    for step_no in step_no_list:
        X = load_X(recipe_no, step_no).T
        print(step_no, X.shape[1]/65)
        X_list.append(X)

    n_data = X_list[0].shape[0]
    maxlen = sum([X.shape[1] for X in X_list])
    
    print('maxlen : {}'.format(maxlen))
    
    row_list = []
    len_list = []
    for i in range(n_data):
        rows = [X[i] for X in X_list]
        rows = [x[~np.isnan(x)] for x in rows]
        row = np.concatenate(rows)
        len_list.append(len(row))
        row = np.pad(row, (0, maxlen - len(row)), 'constant') 
        row_list.append(row)
    X = np.stack(row_list)
    print(X.shape, y.shape)
    len_list = np.array(len_list) / 65
    
    if preprocessing is not None:
        X = preprocessing(X)
    
    if split is True:
        tst_size = 1. - trn_size - val_size
        X_trn, X_tst, y_trn, y_tst, l_trn, l_tst = train_test_split(
            X, y, len_list, test_size=tst_size, random_state=random_state, stratify=y)

        val_size = val_size / (val_size + trn_size)
        X_trn, X_val, y_trn, y_val, l_trn, l_val = train_test_split(
            X_trn, y_trn, l_trn, test_size=val_size, random_state=random_state, stratify=y_trn)

        if scale is True:
            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn = scaler.transform(X_trn)
            X_val = scaler.transform(X_val)
            X_tst = scaler.transform(X_tst)

        return X_trn, X_val, X_tst, y_trn, y_val, y_tst, l_trn, l_val, l_tst

    else:
        return X, y, len_list
    
    



def sampling(maxlen, recipe_data, random_state) :
    np.random.seed(random_state)
    len_ = recipe_data.shape[0]
    plus_choice = np.random.choice(np.arange(0, len(recipe_data)), maxlen - len_)
    output = np.concatenate([recipe_data, recipe_data[plus_choice]])
    index = np.arange(len(output))
    np.random.shuffle(index)
    return output[index]


def resample_data(X, y, i=None, pos_ratio=0.2):
    # i : 길이
    # pos_idx : 아웃라이어
    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]
    
    # 정상 * (정상:비정상 비)
    n_new_pos = int(len(neg_idx) * pos_ratio / (1 - pos_ratio))
    np.random.seed(0)
    # 오버샘플링
    new_pos_idx = np.random.choice(pos_idx, n_new_pos)
    
    X = np.concatenate([X[new_pos_idx], X[neg_idx]], axis=0)
    y = np.concatenate([y[new_pos_idx], y[neg_idx]], axis=0)
    if i is not None:
        i = np.concatenate([i[new_pos_idx], i[neg_idx]], axis=0)
        return X, y, i
    return X, y


class Metric() :
    def __init__(self) :
        self.initialize()
        self.result = {}
        self.result['total'] = 0
        self.result['TP'] = 0
        self.result['TN'] = 0
        self.result['FP'] = 0
        self.result['FN'] = 0
        self.result['label'] = []
        self.result['pred'] = []
        
    def initialize(self) :
        self.total = 0
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_neg = 0
        self.label_list = []
        self.pred_list = []
        
    def measure_metric(self, label, pred) :
        pred_ = pred
        pred = pred.max(1)[1]
        self.label_list += label.cpu().data.tolist()
        self.pred_list += pred_.cpu().data.tolist()
        self.total += label.size()[0]
        self.true_pos += torch.sum(pred * label).item()
        self.false_pos += torch.sum(F.relu(pred - label)).item()
        self.false_neg += torch.sum(F.relu(label - pred)).item()
        self.true_neg = (self.total - self.true_pos - self.false_neg - self.false_pos)
        #print(self.total, self.true_pos, self.false_pos, self.false_neg, self.true_neg)
        
    def get_metric(self) :
        accuracy = (self.result['TP'] + self.result['TN']) / self.result['total'] * 100
        
        if (self.result['TP'] + self.result['FP']) == 0 :
            precision = 0.0
        else :
            precision = self.result['TP'] / (self.result['TP'] + self.result['FP']) *100
        if (self.result['TP'] + self.result['FN']) == 0 :
            recall = 0.0
        else :
            recall = self.result['TP'] / (self.result['TP'] + self.result['FN']) * 100 
        if (precision + recall) == 0:
            F1 = 0.0
        else :
            F1 = 2 * precision * recall / (precision + recall)
            
        pred = [row[1] for row in self.result['pred']]
        AUC = roc_auc_score(self.result['label'], pred)
        
        return accuracy, precision, recall, F1, AUC
    
    def save_metric(self) :
        self.result['total'] += self.total
        self.result['TP'] += self.true_pos
        self.result['TN'] += self.true_neg
        self.result['FP'] += self.false_pos
        self.result['FN'] += self.false_neg
        self.result['label'] += self.label_list
        self.result['pred'] += self.pred_list
        
        

class MyVarDataSet(Dataset):
    def __init__(self, X, y, recipe_no):
        X = self.preprocessing(X)        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.recipe_no = recipe_no

    def preprocessing(self, X): # (timelen * max_features, n_data)
        max_features = 65
        n_data = X.shape[0]
        timelen = int(X.shape[1] / max_features)

        X = X.reshape(n_data, timelen, max_features)
        return X

    def __getitem__(self, index):
        if type(self.recipe_no) is int :
            self.recipe_no = np.array([self.recipe_no for i in range(len(self.X))])
            self.recipe_no = torch.from_numpy(self.recipe_no).long()
        return self.X[index], self.y[index], self.recipe_no[index]

    def __len__(self):
        return len(self.X)


class MultiDataset(Dataset):
    def __init__(self, X1, y1, X2, y2):
        self.X = np.concatenate((X1, X2))
        self.y = np.concatenate((y1, y2)).astype(np.int64)
        self.num = np.array([1] * len(X1) + [2] * len(X2))
        self.X = self.preprocessing(self.X)
        
        index = self.shuffle()
        self.X = self.X[index]
        self.y = self.y[index]
        self.num = self.num[index]

    def preprocessing(self, X): # (timelen * max_features, n_data)
        max_features = 65
        n_data = X.shape[0]
        timelen = int(X.shape[1] / max_features)

        X = X.reshape(n_data, timelen, max_features)
        return X

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.num[index]

    def __len__(self):
        return len(self.X)

    def shuffle(self):
        index = np.arange(0, len(self.X))
        np.random.shuffle(index)
        return index

def train_val_split(kfold, index, y1):
    train_val = StratifiedKFold(n_splits=kfold, shuffle=True)
    recipe1_trn_index, recipe1_val_index = next(iter(train_val.split(index, y1[index])))

    return index[recipe1_trn_index], index[recipe1_val_index]