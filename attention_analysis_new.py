import os, torch, yaml, pickle
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from catbert_prediction import predict_fn
from model.common import backbone_wrapper, checkpoint_loader
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

class AttentionAnalysis():
    def __init__(self, checkpoint_dir, combine_list, data_path, save_path):

        self.checkpoint_dir = checkpoint_dir
        self.combine_list = combine_list
        self.df_raw = pd.read_pickle(data_path).set_index('id')
        self.df_data = self.combine_text(combine_list)
        self.model, self.tokenizer, self.name = self.load_model_and_tokenizer()
        self.save_path = save_path
        #breakpoint()
    
    def load_model_and_tokenizer(self):
        '''
        Load roberta model from checkpoint
        Even though the checkpoint is for the whole model, 
        we only load the backbone to generate the attentions.
        '''
        # load backbone roberta model

        if self.checkpoint_dir == 'roberta-base':
            print('===== Loading roberta-base model =====')
            model = RobertaModel.from_pretrained('roberta-base')
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            name = 'base'
        else:
            print('===== Loading roberta model from checkpoint =====')
            # model_config = yaml.load(open(os.path.join(self.checkpoint_dir, "roberta_config.yaml"), "r"), Loader=yaml.FullLoader)
            # roberta_config = RobertaConfig.from_dict(model_config)
            model = RobertaModel.from_pretrained('roberta-base')
            checkpoint = os.path.join(self.checkpoint_dir, "checkpoint.pt")
            checkpoint_loader(model, checkpoint, load_on_roberta=True)
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            name = checkpoint_dir.split('/')[-1]
        return model, tokenizer, name
    
    def combine_text(self, combine_list):
        
        df = self.df_raw.copy()
        # combine two columns: bulk1, bulk2 --> bulk
        df['bulk'] = df[['bulk1', 'bulk2']].apply(lambda x: ' '.join(x), axis=1)
        df['text'] = df[combine_list].apply(lambda x: ' '.join(x), axis=1)
        
        # drop columns other than 'text' and 'target'
        df = df[['text', 'system', 'adsorbate', 'bulk', 'target']]
        return df

    def predict_attention(self, id):
        '''
        Predict attention for a given id
        '''
        # need to preprocess df_data first! 
        # make df contain the integrated text description
        df = self.df_data[['text', 'target']].loc[id].to_frame().T
        #df = self.df_data.loc[id].to_frame().T
        #breakpoint()
        model = self.model
        tokenizer = self.tokenizer
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        attentions = predict_fn(df, model, tokenizer, device, mode='attn')
        breakpoint()
        return attentions
    

    def get_first_token_idx(self, id, fragment_len=5):
        '''
        Get the index of the first token of each section (system, ads, cat)
        fragment_len: length of the fragment to search for index
        return: dict of first token index for each section
        '''
            
        text = self.df_data.loc[id]['text']
        tokenizer = self.tokenizer
        tokens = tokenizer.encode(text, return_tensors='pt')

        
        sections = {'system': self.df_data['system'].loc[id],
                    'adsorbate': ' ' + self.df_data['adsorbate'].loc[id],
                    'bulk': ' ' + self.df_data['bulk'].loc[id]}
        # breakpoint()
        first_token_ids = {}
        #for key, section in sections.items():
        for key in self.combine_list:
            section = sections[key]
            section_tokens = tokenizer.encode(section, return_tensors='pt')
            fragment = section_tokens[0, 1:1+fragment_len]  # Take the first n tokens
            found = False
            for i in range(0, tokens.size(1) - fragment_len + 1):
                if tokens[0, i:i+fragment_len].equal(fragment):
                    first_token_ids[key] = i  # Index of the first token
                    found = True
                    break    
           
            if not found:
                print(f"Fragment not found for key: {key}")
            
            # result check
            decoded_section = tokenizer.decode(fragment)
            found_section = tokenizer.decode(tokens[0, first_token_ids[key]:first_token_ids[key]+fragment_len])
            if decoded_section != found_section:
                print(f"Decoded section: {decoded_section}")
                print(f"Found section: {found_section}")
                # raise ValueError("Decoded section and found section are not equal")
            else:
                print(f"Success for key: {key}")

        # add eos token index for removing padding
        eos_id = torch.where(tokens[0] == tokenizer.eos_token_id)[0].item()
        first_token_ids['eos'] = eos_id

        return first_token_ids

    def plot_attention_score(self, id):
        '''
        Plot the attention score between <s> and every token in the text for each head
        id: id of the sample
        '''
        attentions = self.predict_attention(id)
        first_token_ids = self.get_first_token_idx(id)
        # breakpoint()
        # Create a 3 by 4 subplot layout
        fig, axes = plt.subplots(3, 4, figsize=(9, 12))
        y_tick_label_dict = {'system': 'sys',
                            'adsorbate': 'ads',
                            'bulk': 'bulk'}
        # Iterate over each subplot
        # implement tqdm to show progress bar
        for i, ax in tqdm.tqdm(enumerate(axes.flatten()), total=12):
            # Get the attention values for the current subplot
            attn = attentions[0, i, 0, :first_token_ids['eos']].detach().cpu().numpy()
            attn = attn.reshape(-1, 1)
            # Plot the heatmap
            sns.heatmap(attn, ax=ax, xticklabels=False, cmap='Reds')#'coolwarm')
            # Set the y-ticks for initial and final points of each section
            if len(self.combine_list) > 1:
                y_ticks = list(first_token_ids.values())
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(['']*len(y_ticks))
                # Set the y-ticks for the middle of each section
                y_ticks_additional = [(y_ticks[i] + y_ticks[i+1]) / 2 for i in range(len(y_ticks) - 1)]
                # y_tick_labels = self.combine_list #['sys', 'conf', 'ads', 'bulk']
                y_tick_labels = [y_tick_label_dict[key] for key in self.combine_list]
                ax.set_yticks(y_ticks_additional, minor=True)
                ax.set_yticklabels(y_tick_labels, minor=True, fontsize=12)
                # Adjust the length of the y-ticks separately
                tick_params = {'length': 15, 'pad': 2}
                ax.tick_params(axis='y', which='major', **tick_params)
                tick_params_minor = {'length':0}
                ax.tick_params(axis='y', which='minor', **tick_params_minor)
            # Set the title for the subplot
            ax.set_title('Head {}'.format(i + 1), fontsize=14)
        # Adjust the layout spacing
        plt.tight_layout()
        # Save the figure
        id = id.split('random')[-1]
        full_save_path = os.path.join(self.save_path, f"{self.name}_{id}.png")
        plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')


if __name__ == '__main__':
    id = "random2534624"
    # example_ids = ['random633147', 'random2190634', 'random2405750',
    #                'random1923577', 'random1943940']
    checkpoint_dir = "checkpoint/finetune/base-sys_ads-ft_0716_2130" #"roberta-base"
    combine_list = ['system', 'adsorbate']
    data_path = "data/df_val_multi2.pkl"
    save_path = "dummy" #"figure/attn"  #base-ft_0623_0038"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    analysis = AttentionAnalysis(checkpoint_dir, combine_list, data_path, save_path)
    analysis.plot_attention_score(id)
    # analysis.plot_section_attention_score(id)