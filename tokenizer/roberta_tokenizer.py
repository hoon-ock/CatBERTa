from pathlib import Path 
from tokenizers import ByteLevelBPETokenizer
import os


# Collect training files for tokenizer
paths = [str(x) for x in Path('../data/train').glob('random*.txt')]

# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, min_frequency=2, 
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


# Save tokenizer
#tokenizer.save('roberta_tokenizer.json')
dir_path = os.getcwd()
tokenizer.save_model(dir_path)