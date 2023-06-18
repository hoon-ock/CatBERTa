import pandas as pd
import numpy as np
import pickle

# determine the number of atoms in the adsorbate
def get_num_atoms(ads):
    num_atoms = 0
    for char in ads:
        if char.isupper():
            num_atoms += 1
        if char.isdigit():
            num_atoms += int(char) - 1
    return num_atoms


# classify the number of atoms into 3 categories
def get_ads_size(ads):
    # 0 (small): 1-2 atoms
    # 1 (medium): 3-5 atoms
    # 2 (large): 6+ atoms
    num_atoms = get_num_atoms(ads)
    if num_atoms <= 2:
        return 0
    elif num_atoms <= 5:
        return 1
    else:
        return 2
    
# classify types of adsorbates into 5 categories
def get_ads_category(ads):
    # 0: O&H
    # 1: C1
    # 2: C2
    # 3: N1
    # 4: N2
    
    # split ads into elements
    elements = []
    for char in ads:
        if char.isupper():
            elements.append(char)
        if char.isdigit():
            elements[-1] += char

    # count the number of C, N, O, H
    num_C = 0
    num_N = 0
    num_O = 0
    num_H = 0
    for element in elements:
        try:
            num = int(element[1])
        except:
            num = 1
        if element[0] == 'C':
            num_C += num
        elif element[0] == 'N':
            num_N += num
        elif element[0] == 'O':
            num_O += num
        elif element[0] == 'H':
            num_H += num
        else:
            print("Error: element not recognized")
            return None
        
    # classify into 5 categories
    if num_N == 2:
        return 4
    elif num_N == 1:
        return 3
    elif num_C == 2:
        return 2
    elif num_C == 1:
        return 1
    elif num_O >= 1 or num_H >=1:
        return 0

def add_class_labels(df, metadata):
    df['ads'] = np.array([metadata[id]['ads_symbols'].replace('*', '') for id in df['id']])
    df['ads_size'] = df['ads'].apply(get_ads_size)
    df['ads_class'] = df['ads'].apply(get_ads_category)
    df['bulk_class'] = np.array([metadata[id]['class'] for id in df['id']])
    df.drop(columns=['ads'], inplace=True)
    return df     
    
if __name__ == "__main__":
    # load data
    df_train = pd.read_pickle("df_is2re_100k.pkl")
    df_val = pd.read_pickle("df_is2re_val_25k.pkl")
    metadata = pickle.load(open("../metadata/oc20_data_mapping.pkl", "rb"))
    # for debugging
    df_train = df_train.iloc[:1000]
    df_val = df_val.iloc[:1000]
    
    # add labels
    df_train = add_class_labels(df_train, metadata)
    df_val = add_class_labels(df_val, metadata)

    # save data
    df_train.to_pickle("../data/df_train_w_class.pkl")
    df_val.to_pickle("../data/df_val_w_class.pkl")