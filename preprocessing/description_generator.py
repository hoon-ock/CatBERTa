import pickle
import os
from tqdm import tqdm


# pickle files
read_path = '/home/jovyan/shared-scratch/jhoon/CATBERT/train_prompt_dict/'
save_path = '../data/train/' #read_path+'text/'

files = os.listdir(read_path)
codes = [f.split('.pkl')[0] for f in files if '.pkl' in f]
for code in tqdm(codes):
    save_file_path = os.path.join(save_path, f"{code}.txt")
    if os.path.exists(save_file_path):
        # print(f'{code} already exists')
        continue
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
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(save_file_path, 'w') as file:
        file.write(text)

print('text conversion completed!')    