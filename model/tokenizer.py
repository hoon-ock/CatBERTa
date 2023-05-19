from tokenizers import ByteLevelBPETokenizer
import os

# Define the path to your text files folder
path = "../data/train/"

# # Initialize a new tokenizer
# tokenizer = ByteLevelBPETokenizer()

# # Train the tokenizer on your text files
# tokenizer.train(files=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')],
#                 special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

# # Save the tokenizer to disk
# tokenizer.save("CAT_tokenizer2.json", pretty = True)


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a new tokenizer
tokenizer = Tokenizer(BPE())

# Set up the trainer
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

# Add your text files to the tokenizer
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(files=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')], trainer=trainer)
tokenizer.save("CAT_tokenizer.json")