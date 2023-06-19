from pathlib import Path 
from tokenizers import ByteLevelBPETokenizer
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace
import os
import pandas as pd

file_path = os.path.abspath(__file__)
home_path = os.path.dirname(os.path.dirname(file_path))
#save_path = os.path.join(home_path, 'data', 'df_train.pkl')

# Collect training files for tokenizer
# paths = [str(x) for x in Path('../data/train').glob('random*.txt')]

# Collect training text from dataframe
df = pd.read_pickle('../data/df_train.pkl')
texts = df['text'].values.tolist()
print('Number of training texts: ', len(texts))


# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()
# tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(texts)
tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
# tokenizer.train(files=paths, min_frequency=2, 
#                 special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


# Save tokenizer
#tokenizer.save('roberta_tokenizer.json')
dir_path = os.getcwd()
tokenizer.save_model(dir_path)