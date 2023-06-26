import os, torch, yaml, pickle
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from catbert_prediction import predict_fn
from model.common import backbone_wrapper, checkpoint_loader
import seaborn as sns
import matplotlib.pyplot as plt

class AttentionAnalysis():
    def __init__(self, checkpoint_dir, data_path, save_path):

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_dir.split('/')[-1]
        self.df_data = pd.read_pickle(data_path).set_index('id')
        self.model_config = yaml.load(open(os.path.join(self.checkpoint_dir, "roberta_config.yaml"), "r"), Loader=yaml.FullLoader)
        self.model = self.load_model()
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.max_len = self.model_config['max_position_embeddings']
        self.tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer', max_len=self.max_len)
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=self.max_len)
        self.save_path = save_path

    
    def load_model(self):
        '''
        Load roberta model from checkpoint
        Even though the checkpoint is for the whole model, 
        we only load the backbone to generate the attentions.
        '''
        # load backbone roberta model
        roberta_config = RobertaConfig.from_dict(self.model_config)
        backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)

        # load checkpoint
        checkpoint = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        model = checkpoint_loader(backbone, checkpoint, load_on_roberta=True)
        return model

    def predict_attention(self, id, base_mode=False):
        '''
        Predict attention for a given id
        '''
        df = self.df_data.loc[id].to_frame().T
        if base_mode:
            model = self.roberta_model
            tokenizer = self.roberta_tokenizer
        else:
            model = self.model
            tokenizer = self.tokenizer
        device = torch.device("cpu")
        attentions = predict_fn(df, model, tokenizer, device, mode='attn')
        return attentions
    

    def get_first_token_idx(self, id, fragment_len=10):
        '''
        Get the index of the first token of each section (system, conf_i, ads, cat)
        fragment_len: length of the fragment to search for index
        return: dict of first token index for each section
        '''
        # load the prompt files
        path = "/home/jovyan/shared-scratch/jhoon/CATBERT/new_prompt"
        file = pickle.load(open(os.path.join(path, f"{id}.pkl"), "rb"))
        text = self.df_data.loc[id]['text']
        tokenizer = self.tokenizer
        tokens = tokenizer.encode(text, return_tensors='pt')

        
        sections = {'system': file['system'],
                    'conf_i': ' '+file['conf_i'],
                    'ads': file['ads'],
                    'cat': file['cat']}
        
        first_token_ids = {}
        for key, section in sections.items():
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
                raise ValueError("Decoded section and found section are not equal")
            else:
                print(f"Success for key: {key}")

        # add eos token index for removing padding
        eos_id = torch.where(tokens[0] == tokenizer.eos_token_id)[0].item()
        first_token_ids['eos'] = eos_id

        return first_token_ids

    def plot_attention_score(self, id, base_mode=False):
        '''
        Plot the attention score between <s> and every token in the text for each head
        id: id of the sample
        base_mode: whether to use base roberta model or fine-tuned model
        '''
        attentions = self.predict_attention(id, base_mode)
        first_token_ids = self.get_first_token_idx(id)
        # Create a 3 by 4 subplot layout
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))

        # Iterate over each subplot
        for i, ax in enumerate(axes.flatten()):
            # Get the attention values for the current subplot
            attn = attentions[0, i, 0, :first_token_ids['eos']].detach().cpu().numpy()
            attn = attn.reshape(-1, 1)

            # Plot the heatmap
            sns.heatmap(attn, ax=ax, xticklabels=False, cmap='Reds')#'coolwarm')

            # Set the title for the subplot
            ax.set_title('Head {}'.format(i + 1))

        # Adjust the layout spacing
        plt.tight_layout()

        # Save the figure
        if base_mode:
            prefix='base-'
        else:
            prefix= self.checkpoint_name
        id = id.split('random')[-1]
        full_save_path = os.path.join(self.save_path, f"{prefix}_attn_{id}.png")
        plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')

    def plot_section_attention_score(self, id, base_mode=False, relative=True, combined=False):
        '''
        Aggregate the attention score for each section and plot the score
        id: id of the sample
        base_mode: whether to use base roberta model or finetuned model
        relative: whether to plot the averaged score or summed score
        combined: whether to plot the combined score for all heads (under construction)
        '''
        attentions = self.predict_attention(id, base_mode)
        first_token_ids = self.get_first_token_idx(id)
        # convert first_token_ids to list of tuples
        # [(system, conf_i), (conf_i, ads), (ads, cat), (cat, eos)]
        keys = list(first_token_ids.keys())
        index_range = [(first_token_ids[keys[i]], first_token_ids[keys[i+1]]) for i in range(len(keys)-1)]
        # Create a 3 by 4 subplot layout
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        combined_attn = np.empty((4, 0))
        # Iterate over each subplot
        for i, ax in enumerate(axes.flatten()):
            # Get the attention values for the current subplot
            attn = attentions[0, i, 0, :first_token_ids['eos']].detach().cpu().numpy()
            attn = attn.reshape(-1, 1)

            section_attn = np.zeros((4, 1))
            for j, indices_range in enumerate(index_range):  
                if relative:
                    section_attn[j] = np.mean(attn[indices_range[0]:indices_range[1]])
                else:
                    section_attn[j] = np.sum(attn[indices_range[0]:indices_range[1]])
            # Plot the heatmap
            sns.heatmap(section_attn, ax=ax, xticklabels=False, cmap='Reds')

            # Set the title and y-tick labels for the subplot
            ax.set_title('Head {}'.format(i + 1), fontsize=14)

            # Calculate the y-tick positions at the center of each block
            y_ticks = np.arange(4) + 0.5
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(['sys', 'conf', 'ads', 'bulk'], rotation=15, fontsize=12)

            combined_attn = np.concatenate((combined_attn, section_attn), axis=1)
        # Adjust the layout spacing
        plt.tight_layout()
        if base_mode:
            prefix='base-'
        else:
            prefix= self.checkpoint_name
        id = id.split('random')[-1]
        full_save_path = os.path.join(self.save_path, f"{prefix}_attn_section_{id}.png")
        plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
        
        if combined:
            print('This option is not implemented yet')
            fig_combined, ax_combined = plt.subplots(figsize=(12, 9))
            sns.heatmap(combined_attn, ax=ax_combined, cmap='Reds')
            full_save_path = os.path.join(self.save_path, f"attn_section_{id}.png")
            plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')




if __name__ == '__main__':
    id = "random2534624"
    # example_ids = ['random633147', 'random2190634', 'random2405750',
    #                'random1923577', 'random1943940']
    checkpoint_dir = "checkpoint/pretrain/mlm-pt_0623_0522"
    data_path = "data/df_val.pkl"
    save_path = "results/dummy/"

    analysis = AttentionAnalysis(checkpoint_dir, data_path, save_path)
    analysis.plot_attention_score(id, base_mode=True)
    analysis.plot_section_attention_score(id, base_mode=True)