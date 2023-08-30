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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.split_names = ['id', 'ood_ads', 'ood_cat', 'ood_both']
        self.df_emb = self.get_df_emb()
        self.tsne_obj = self.run_tsne(perplexity)
        self.label_mapping = {
                            'split': {0: 'ID', 1: 'OOD$_{ads}$', 2: 'OOD$_{cat}$', 3: 'OOD$_{both}$'},
                            'ads_size': {0: 'Small', 1: 'Medium', 2: 'Large'},
                            'ads_class': {0: 'O&H', 1: 'C1', 2: 'C2', 3: 'N1', 4: 'N2'},
                            'bulk_class': {0: 'Intermetallic', 1:'Metalloid', 2:'Non-metal', 3:'Halids'},
                            'space_group': {0: 'Fm-3m', 1: 'P6_3/mmc', 2: 'Pnma', 3: 'Pm-3m', 4: "Others"},
                            'bulk_type': {0: 'Zr', 1: 'Al', 2: 'Ni', 3: 'Others'}
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
        df_emb['bulk_type'] = self.df.loc[df_emb.index, 'bulk_type']
        #df_emb['space_group'] = self.df.loc[df_emb.index, 'space_group']
        #breakpoint()
        # Check the validity of the labels
        if not np.array_equal(np.unique(df_emb['split']), [0, 1, 2, 3]):
            raise ValueError("splits are not valid")
        if not np.array_equal(np.unique(df_emb['ads_size']), [0, 1, 2]):
            raise ValueError("ads_size are not valid")
        if not np.array_equal(np.unique(df_emb['ads_class']), [0, 1, 2, 3, 4]):
            raise ValueError("ads_class are not valid")
        if not np.array_equal(np.unique(df_emb['bulk_class']), [0, 1, 2, 3]):
            raise ValueError("bulk_class are not valid")
        if not np.array_equal(np.unique(df_emb['bulk_type']), [0, 1, 2, 3]):
            raise ValueError("bulk_type are not valid")
        # if not np.array_equal(np.unique(df_emb['space_group']), [0, 1, 2, 3, 4]):
        #     raise ValueError("space_group are not valid")
        return df_emb
    
    def run_tsne(self, perplexity=10):
        X = self.df_emb.drop(columns=['split', 'ads_size', 
                                      'ads_class', 'bulk_class', 
                                      'bulk_type']).values
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
        fig, ax = plt.subplots(frameon=False)
        #breakpoint()
        cmap = sns.color_palette("viridis", as_cmap=True)
        sns.scatterplot(
            x='tsne_1', y='tsne_2',
            palette=cmap,
            hue='label',
            # palette=sns.color_palette("hls", 4),
            data = tsne_result,
            ax=ax,
            legend = "full",
            alpha=0.5
        )
        lim = (self.tsne_obj.min()-10, self.tsne_obj.max()+10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Update the legend labels
        legend_labels =  list(self.label_mapping[target_label].values()) # [label_mapping[label][val] for val in sorted(label_mapping[label].keys())]
        handles, _ = ax.get_legend_handles_labels()
        # legend = ax.legend(handles, legend_labels, 
        #                     loc= 'upper left', fontsize=13)
        # if "ads" in target_label:
        #     num = 3
        # else:
        #     num = 2
        num=1
        legend = ax.legend(handles, legend_labels, loc='lower right', 
                           fontsize=15, ncol=num)

        # Make the legend box transparent
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 0, 0))

        # Set axis labels
        # ax.set_xlabel("t-SNE 1", fontsize=16)
        # ax.set_ylabel("t-SNE 2", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Adjust spacing and padding
        plt.tight_layout()  

        # # Calculate the center coordinates of the scatter plot
        # scatter_center_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        # scatter_center_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        # #Get the width and height of the legend box
        # legend_width, legend_height = legend.get_frame().get_width(), legend.get_frame().get_height()

        # # Adjust the legend position to align with the scatter plot center
        # legend.set_bbox_to_anchor((scatter_center_x - legend_width / 2, scatter_center_y + legend_height / 2))

        
        full_save_path = os.path.join(self.save_dir, f"{target_label}.png")
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight', facecolor='w')
    
if __name__ == "__main__":
    data_path = '/home/jovyan/shared-scratch/jhoon/CATBERT/is2re/val_2nd/val_10k_label.pkl'
    emb_path = 'results/embed/catbert_base.pkl' #catbert_ft_100k_1.pkl'
    perplexity = 30
    tag = emb_path.split('/')[-1].split('catbert_')[-1].split('.pkl')[0]
    save_dir = f'figure/embed/no_frame/{tag}'
    emb_analysis = EmbeddingAnalysis(data_path, emb_path, perplexity, save_dir)
    
    for label in ['split', 'ads_size', 'ads_class', 'bulk_type', 'bulk_class']:
        print(f"Plotting {label}...!")
        emb_analysis.plot_tsne(label)