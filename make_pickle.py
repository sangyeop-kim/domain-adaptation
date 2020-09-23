import pickle
from utils import *

step_list = list(range(10,17))

recipe_no = 1
X1, y1, l1 = load_data_cat_nan(recipe_no, step_list,# 14, 15],
                            preprocessing=None, split=False, scale=False)


recipe_no = 2
X2, y2, l2 = load_data_cat_nan(recipe_no, step_list,# 14, 15],
                            preprocessing=None, split=False, scale=False)

data = {}
data['X1'] = X1
data['X2'] = X2
data['y1'] = y1
data['y2'] = y2
data['l1'] = l1
data['l2'] = l2

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
    print('completely made pickle!')
    