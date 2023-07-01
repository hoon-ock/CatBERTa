import numpy as np
import pandas as pd
import os, pickle

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ==========================================================
# This script is to obtain the t-SNE plot of the embeddings
# ==========================================================

class EmbeddingAnalysis:
    def __init__(self, data_path, emb_path, perplexity, save_dir):
        '''
        data_path: path for full data with keys: 'id', 'text', 'label', 'class'
        emb_path: path for embeddings of data (subset or full)
        perplexity: perplexity for t-SNE
        save_dir: plot save directory
        '''
        self.df = pd.read_pickle(data_path)
        self.emb_path = emb_path
        self.perplexity = perplexity
        self.save_dir = save_dir
        self.split_names = ['id', 'ood_ads', 'ood_cat', 'ood_both']
        self.df_emb = self.get_df_emb()
        self.tsne_obj = self.run_tsne(perplexity)
        self.label_mapping = {
                            'split': {0: 'ID', 1: 'OOD_ads', 2: 'OOD_cat', 3: 'OOD_both'},
                            'ads_size': {0: 'Small', 1: 'Medium', 2: 'Large'},
                            'ads_class': {0: 'O&H', 1: 'C1', 2: 'C2', 3: 'N1', 4: 'N2'},
                            'bulk_class': {0: 'Intermetallic', 1:'Metalloid', 2:'Non-metal', 3:'Halids'}
                            }


    def get_split_ids(self):
        '''
        splits: ID, OOD-ads, OOD-cat, OOD-both
        '''
        split_id_path = 'metadata/split_ids'
        
        split_ids = {}
        for split in self.split_names:
            path = os.path.join(split_id_path, f"full_normal_{split}.pkl")
            with open(path, 'rb') as f:
                split_ids[split] = pickle.load(f)
        return split_ids

    
    def get_df_emb(self):
        '''
        obtain df_emb from embeddings of roberta model
        return: df_emb with columns: 'split', 'ads_size', 'ads_class', 'bulk_class'
        '''
        # Set id as index for dataset
        self.df.set_index('id', inplace=True)
        # Load embeddings
        emb = pd.read_pickle(self.emb_path)
        df_emb = pd.DataFrame.from_dict(emb, orient='index')
        # Set split labels for embeddings
        split_mapping = {'id': 0, 'ood_ads': 1, 'ood_cat': 2, 'ood_both': 3}
        split_ids = self.get_split_ids()
        for id in df_emb.index:
            for split in self.split_names:
                if id in split_ids[split]:
                    df_emb.loc[id, 'split'] = split_mapping[split]
        # Set other labels for embeddings
        df_emb['ads_size'] = self.df.loc[df_emb.index, 'ads_size']
        df_emb['ads_class'] = self.df.loc[df_emb.index, 'ads_class']
        df_emb['bulk_class'] = self.df.loc[df_emb.index, 'bulk_class']
        # Check the validity of the labels
        if not np.array_equal(np.unique(df_emb['split']), [0, 1, 2, 3]):
            raise ValueError("splits are not valid")
        if not np.array_equal(np.unique(df_emb['ads_size']), [0, 1, 2]):
            raise ValueError("ads_size are not valid")
        if not np.array_equal(np.unique(df_emb['ads_class']), [0, 1, 2, 3, 4]):
            raise ValueError("ads_class are not valid")
        if not np.array_equal(np.unique(df_emb['bulk_class']), [0, 1, 2, 3]):
            raise ValueError("bulk_class are not valid")
        return df_emb
    
    def run_tsne(self, perplexity=10):
        X = self.df_emb.drop(columns=['split', 'ads_size', 
                                      'ads_class', 'bulk_class']).values
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=0)
        tsne_obj = tsne.fit_transform(X)
        return tsne_obj
        
    
    def plot_tsne(self, target_label):
        '''
        target_label: 'split', 'ads_size', 'ads_class', 'bulk_class'
        '''
        y = self.df_emb[target_label].values
        # t-SNE  
        tsne_result = pd.DataFrame({'tsne_1': self.tsne_obj[:,0], 
                                    'tsne_2': self.tsne_obj[:,1], 
                                    'label': y})

        # plot
        fig, ax = plt.subplots(1)
        #breakpoint()
        cmap = sns.color_palette("Set1", as_cmap=True)
        sns.scatterplot(
            x='tsne_1', y='tsne_2',
            palette=cmap,
            hue='label',
            # palette=sns.color_palette("hls", 4),
            data = tsne_result,
            ax=ax,
            legend = "full",
            alpha=0.2
        )
        lim = (self.tsne_obj.min()-5, self.tsne_obj.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Update the legend labels
        legend_labels =  list(self.label_mapping[target_label].values()) # [label_mapping[label][val] for val in sorted(label_mapping[label].keys())]
        handles, _ = ax.get_legend_handles_labels()
        legend = ax.legend(handles, legend_labels, 
                            loc='lower left', fontsize=14)

        # Make the legend box transparent
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 0, 0))

        # Set axis labels
        ax.set_xlabel("t-SNE 1", fontsize=16)
        ax.set_ylabel("t-SNE 2", fontsize=16)

        # Adjust spacing and padding
        plt.tight_layout()  
        
        full_save_path = os.path.join(self.save_dir, f"{target_label}.png")
        plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
    
if __name__ == "__main__":
    data_path = 'data/df_val.pkl'
    emb_path = 'results/embed/catbert_embed_base-ft_0623_0038.pkl'
    perplexity = 30
    save_dir = 'figure/embed/base-ft_0623_0038'
    emb_analysis = EmbeddingAnalysis(data_path, emb_path, perplexity, save_dir)
    
    for label in ['split', 'ads_size', 'ads_class', 'bulk_class']:
        print(f"Plotting {label}...!")
        emb_analysis.plot_tsne(label)