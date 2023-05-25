import pickle
import os
from tqdm import tqdm
import pandas as pd

file_path = os.path.abspath(__file__)
home_path = os.path.dirname(os.path.dirname(file_path))
save_path = os.path.join(home_path, 'data', 'df_train.pkl')

# read pickle files
read_path = '/home/jovyan/shared-scratch/jhoon/CATBERT/train_prompt_dict/'
files = os.listdir(read_path)
codes = [f.split('.pkl')[0] for f in files if '.pkl' in f]

# loop over files to obtain text and energy
dict = {}
code_list, text_list, E_list = [], [], []
for code in tqdm(codes):
    result = pickle.load(open(read_path+code+'.pkl', 'rb'))
    sys = result['system']
    ads = result['ads']
    cat = result['cat']
    conf_i = result['conf_i']
    conf_f = result['conf_f'].split(' The')[0]
    E = result['E_ads']
    ## check if the data is valid
    if "" in result.values():
        print(f'{code} has an empty value')
        continue
    if 'Error' in result.values():
        print(f'{code} has an error')
        continue
    text = sys + ' ' + conf_f + '\n\n' + ads + '\n\n' + cat
    code_list.append(code)
    text_list.append(text)
    E_list.append(E)

dict = {'id': code_list, 'text': text_list, 'target': E_list}
df = pd.DataFrame(dict)

# save dataframe
with open(save_path, 'wb') as f:
    pickle.dump(df, f)