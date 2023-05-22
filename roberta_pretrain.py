from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM
from pathlib import Path 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm 
import wandb
import datetime

# mask language modeling
def mlm(tensor, mask_token_id=4):
    rand = torch.rand(tensor.shape) #[0,1]
    mask_arr = (rand < 0.15) * (tensor > 2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero())
        tensor[i, selection] = mask_token_id #tokenizer.mask_token_id
    return tensor 

# convert files to input tensors
def get_encodings(paths):
    input_ids = []
    mask = []
    labels = []
    for path in tqdm(paths):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        sample = tokenizer(text, max_length=512, padding='max_length', 
                           truncation=True, return_tensors='pt')
        labels.append(sample['input_ids'])
        mask.append(sample['attention_mask'])
        input_ids.append(mlm(sample['input_ids'].clone()))
    input_ids = torch.cat(input_ids)
    mask = torch.cat(mask)
    labels = torch.cat(labels)
    return {'input_ids': input_ids, 
            'attention_mask': mask, 
            'labels': labels}

# train function
def train(model, dataloader, optim, device):    
    model.train()
    loop = tqdm(dataloader, leave=True)
    total_loss = 0
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        total_loss += loss.item()
        # loop.set_description('Epoch: {}'.format(epoch + 1))
        # loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

# roberta dataset class
class RobertaDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])


if __name__ == '__main__':
    
    # hyperparameters for training
    num_epochs = 5
    lr = 1e-4
    batch_size = 32

    # load pre-trained tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer', max_len=514)

    # load training files
    paths = [str(x) for x in Path('./data/train').glob('random*.txt')]
    encodings = get_encodings(paths)
    dataset = RobertaDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load config and model
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    model.to(device)
    optim = AdamW(model.parameters(), lr=lr)
    
    # set up wandb
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run{now}"
    wandb.init(project='catbert-test', name=run_name, dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    wandb.watch(model, log="all")


    # training loop
    for epoch in range(num_epochs):
        loss = train(model, dataloader, optim, device)
        print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")
        #model.save_pretrained('model'
        if epoch%2 == 0:
            wandb.log({"loss": loss})
    
    torch.save(model.state_dict(), f'./checkpoint/pretrain/{run_name}.pt')
    wandb.finish()

