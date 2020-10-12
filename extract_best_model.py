from collections import Counter
from glob import glob
import pandas as pd
import numpy as np
import argparse
import warnings
import shutil
warnings.filterwarnings('ignore')


def delete_others():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()
    continue_training = args.train
    
    performance = glob('./best_model/**/*.txt')
    df1 = pd.DataFrame(columns = ['path', 'model', 'attn_kernel', 'encoding_kernel', 'F1'])
    df2 = pd.DataFrame(columns = ['path', 'model', 'attn_kernel', 'encoding_kernel', 'F1'])

    for per in performance:
        with open(per, 'r') as f:
            file = f.read().split('\n')
            recipe = [f for f in file if 'recipe' in f]
            file = [f for f in file if 'recipe' not in f]
            
            df1 = pd.concat((df1, pd.DataFrame({'path' : [per.split('/')[2]], 'model' : [file[0]], 
                                            'encoding_kernel' : [int(file[1].split(' : ')[1])],
                                            'attn_kernel' : [int(file[2].split(' : ')[1])],
                                            'F1' : [float(recipe[0].split('F1 : ')[1]\
                                                            .split(', ')[0])]})), 
                        axis=0)
            
            if len(recipe) > 1:

                df2 = pd.concat((df2, pd.DataFrame({'path' : [per.split('/')[2]], 'model' : [file[0]], 
                                                'encoding_kernel' : [int(file[1].split(' : ')[1])],
                                                'attn_kernel' : [int(file[2].split(' : ')[1])],
                                                'F1' : [float(recipe[1].split('F1 : ')[1]\
                                                                .split(', ')[0])]})), 
                            axis=0)


    rank1 = np.array(df1.groupby(['model', 'attn_kernel', 'encoding_kernel']))

    sota1 = {}
    del_list1 = []

    for num, _ in enumerate(rank1):
        temp_df = rank1[num][1].sort_values('F1', ascending=False)
        sota1[rank1[num][0]] = temp_df[:3]
        del_list1 += temp_df['path'][3:].tolist()
        
        
    sota2 = {}
    del_list2 = []
        
    if df2.shape[0] > 0:
        rank2 = np.array(df2.groupby(['model', 'attn_kernel', 'encoding_kernel']))

        
        for num, _ in enumerate(rank2):
            temp_df = rank2[num][1].sort_values('F1', ascending=False)
            sota2[rank2[num][0]] = temp_df[:3]
            del_list2 += temp_df['path'][3:].tolist()
    
    if len(del_list2) > 0: 
        index = np.array(list(dict(Counter(del_list1 + del_list2)).values())) != 1
        del_list = np.array(list(dict(Counter(del_list1 + del_list2))))[index].tolist()
        del_list += [i for i in del_list1 if 'single' in i]
    else:
        del_list = del_list1
    
    
    for i in del_list:
        try:

            shutil.rmtree('./best_model/%s' % i)
            print('delete : ./best_model/%s' % i)
        except:
            continue
            
            
    sota_file = [i.split('/perfor')[0] for i in glob('./best_model/**/*.txt')]
    none_txt_list = [i for i in glob('./best_model/*') if i not in sota_file]
    
    if not continue_training:
        for i in none_txt_list:
            try:
                shutil.rmtree(i)
            except:
                continue

    for k,v in sota1.items():
        print(k, v['F1'].values)

    print()
        
    for k,v in sota2.items():
        print(k, v['F1'].values)

        
if __name__=="__main__":
	delete_others()