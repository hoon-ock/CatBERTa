import pandas as pd
import pickle
import matplotlib.pyplot as plt
import glob
from figure.plots import parity_plot, get_array_for_grouping, grouping_fast
import os

class SplitAnalysis():

    def __init__(self, df_train, df_val, metadata, save_path):
        self.df_train = df_train
        self.df_val = df_val
        self.metadata = metadata
        self.save_path = save_path
        self.title_map = {'cgcnn': 'CGCNN',
                          'dimenet': 'DimeNet',
                          'schnet': 'SchNet',
                          'dimenetpp': 'DimeNet++',
                          'painn': 'PaiNN',
                          'catbert': 'CatBERT',}
        self.split_names = ['ID', 'OOD$_{ads}$', 'OOD$_{cat}$', 'OOD$_{both}$']
        self.name_map = {'ID': 'ID',
                        'OOD$_{ads}$': 'OOD-ads',
                        'OOD$_{cat}$': 'OOD-cat',
                        'OOD$_{both}$': 'OOD-both'}
        
        self.ads_list, self.bulk_list = self.system_analysis()
        self.ID, self.OOD_ads, self.OOD_cat, self.OOD_both = self.split_groups()
    
    def system_analysis(self):
        ads_list, bulk_list = [], []
        for i in self.df_train['id']:
            ads_list.append(self.metadata[i]['ads_symbols'])
            bulk_list.append(self.metadata[i]['bulk_symbols'])
        ads_list = list(set(ads_list))
        bulk_list = list(set(bulk_list))
        print("------ System Analysis (train) ------")
        print("Number of adsorbates: " + str(len(ads_list)))
        print("Number of bulks: " + str(len(bulk_list)))
        print("Number of systems: " + str(len(self.df_train)))

        ads_list_val, bulk_list_val = [], []
        for i in self.df_val['id']:
            ads_list_val.append(self.metadata[i]['ads_symbols'])
            bulk_list_val.append(self.metadata[i]['bulk_symbols'])
        ads_list_val = list(set(ads_list_val))
        bulk_list_val = list(set(bulk_list_val))
        print("------ System Analysis (val) ------")
        print("Number of adsorbates: " + str(len(ads_list_val)))
        print("Number of bulks: " + str(len(bulk_list_val)))
        print("Number of systems: " + str(len(self.df_val)))

        return ads_list, bulk_list

    def split_groups(self):
        ads_list_train = self.ads_list
        bulk_list_train = self.bulk_list
        ID, OOD_ads, OOD_cat, OOD_both = [], [], [], []
        for id in self.df_val['id']:
            ads = self.metadata[id]['ads_symbols']
            bulk = self.metadata[id]['bulk_symbols']

            if ads not in ads_list_train and bulk not in bulk_list_train:
                OOD_both.append(id)
            elif ads not in ads_list_train:
                OOD_ads.append(id)
            elif bulk not in bulk_list_train:
                OOD_cat.append(id)
            else:
                ID.append(id)
        print("--------- Split Analysis ---------")
        print("Number of ID: " + str(len(ID)))
        print("Number of OOD_ads: " + str(len(OOD_ads)))
        print("Number of OOD_cat: " + str(len(OOD_cat)))
        print("Number of OOD_both: " + str(len(OOD_both)))
        return ID, OOD_ads, OOD_cat, OOD_both
    
    def get_ml_and_dft_results(self, model, size):

        df = self.df_val.set_index('id')
        dft = self.df_val['target']
        file_path = f"/home/jovyan/CATBERT/data/ml-pred/val_{model}_{size}.pkl"
        result = pd.read_pickle(file_path)
        ml = pd.Series(result)
        dft = df['target']
        df_combined = pd.concat([dft, ml], axis=1)
        df_combined.columns = ['dft', 'ml']
        return df_combined
    
    # def plot_entire_val(self, model, size, plotoff=False):

    #     df = self.get_ml_and_dft_results(model, size)

    #     r2, mae, rmse = parity_plot(df['dft'], df['ml'],
    #                                 xlabel='DFT $\Delta E$ [eV]',
    #                                 ylabel=f'{self.title_map[model]} $\Delta E$ [eV]',
    #                                 plot_type='hexabin', xylim=[-12, 12])
    #     plt.title('Entire data', fontsize=20)
    #     full_save_path = os.path.join(self.save_path, f'{model}_{size}_entire.png')
    #     plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
    #     return r2, mae, rmse
    
    def create_save_directory(self, model, size):
        save_path = os.path.join(self.save_path, f'{model}_{size}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def plot_val_splits(self, model, size):
        save_path = self.create_save_directory(model, size)

        results = {}
        df = self.get_ml_and_dft_results(model, size)
        r2, mae, rmse = parity_plot(df['dft'], df['ml'],
                                    xlabel='DFT $\Delta E$ [eV]',
                                    ylabel=f'{self.title_map[model]} $\Delta E$ [eV]',
                                    plot_type='hexabin', xylim=[-12, 12])
        plt.title('Entire data', fontsize=20)
        full_save_path = os.path.join(save_path, f'entire.png')
        plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
        plt.close()

        results['entire'] = [r2, mae, rmse]  

        groups = [self.ID, self.OOD_ads, self.OOD_cat, self.OOD_both]
        #names = ['ID', 'OOD$_{ads}$', 'OOD$_{cat}$', 'OOD$_{both}$']

        
        for group, name in zip(groups, self.split_names):
            dft = df.loc[group]['dft']
            ml = df.loc[group]['ml']
            r2, mae, rmse = parity_plot(dft, ml,
                                        xlabel='DFT $\Delta E$ [eV]',
                                        ylabel=f'{self.title_map[model]} $\Delta E$ [eV]',
                                        plot_type='hexabin', xylim=[-12, 12])
            plt.title(name, fontsize=20)
            #full_save_path = os.path.join(self.save_path, f'{model}_{size}_{name}.png')
            full_save_path = os.path.join(save_path, f'{self.name_map[name]}.png')
            plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
            plt.close()
            results[self.name_map[name]] = [r2, mae, rmse]

        # with open(os.path.join(self.save_path, f'{model}_{size}.pkl'), 'wb') as f:
        #     pickle.dump(results, f)
        df_results = pd.DataFrame(results).T
        df_results.columns = ['r2', 'mae', 'rmse']
        print(df_results)
        df_results.to_csv(os.path.join(save_path, f'{model}_{size}.csv'))
        return df_results
    
    def plot_energy_difference(self, model, size, 
                               sample_num=400, random_seed=17):
        save_path = self.create_save_directory(model, size)

        # grouping chemically similar pairs 
        file_path = f"/home/jovyan/CATBERT/data/ml-pred/val_{model}_{size}.pkl"
        ml_result = pd.read_pickle(file_path)
        df = self.df_val.set_index('id')

        groups = [self.ID, self.OOD_ads, self.OOD_cat, self.OOD_both]
        names = ['ID', 'OOD$_{ads}$', 'OOD$_{cat}$', 'OOD$_{both}$']
        results = {}
        for group, name in zip(groups, names):

            df_sample = df.loc[group].sample(n=sample_num, random_state=random_seed)
            df_sample = df_sample.reset_index()
            print("============ " + name + " ============")
            code, dft, ml, ads_id, bulk_id = get_array_for_grouping(df_sample, self.metadata, ml_result)
            cat_swap, ads_swap, conf_swap, all_swap = grouping_fast(code, dft, ml, ads_id, bulk_id)
            all_pairs = {**cat_swap, **ads_swap, **conf_swap, **all_swap}
            df_total = pd.DataFrame.from_dict(all_pairs, orient='index', columns=['dft', 'ml'])
            _, _, rmse_t = parity_plot(df_total['dft'], df_total['ml'],
                                        xlabel='DFT $\Delta \Delta E$ [eV]',
                                        ylabel=f'{self.title_map[model]} $\Delta \Delta E$ [eV]',
                                        plot_type='hexabin', xylim=[-12, 12])
            plt.title(f"Entire {name}", fontsize=20)
            #full_save_path = os.path.join(self.save_path, f'ddE_{model}_{size}_{name}_entire.png')
            full_save_path = os.path.join(save_path, f'ddE_{self.name_map[name]}_entire.png')
            plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
            del df_total, all_pairs
            plt.close()
            # plot parity plot
            similar_pairs = {**cat_swap, **ads_swap, **conf_swap}
            df_result = pd.DataFrame.from_dict(similar_pairs, orient='index', columns=['dft', 'ml'])
            r2, mae, rmse = parity_plot(df_result['dft'], df_result['ml'],
                                        xlabel='DFT $\Delta \Delta E$ [eV]',
                                        ylabel=f'{self.title_map[model]} $\Delta \Delta E$ [eV]',
                                        plot_type='hexabin', xylim=[-12, 12])
            plt.title(name, fontsize=20)
            #full_save_path = os.path.join(self.save_path, f'ddE_{model}_{size}_{name}_similar.png')
            full_save_path = os.path.join(save_path, f'ddE_{self.name_map[name]}_similar.png')
            plt.savefig(full_save_path, bbox_inches='tight', facecolor='w')
            plt.close()
 
            secr = 100*(1-rmse/rmse_t)
            results[self.name_map[name]] = [r2, mae, rmse, rmse_t, secr]

        # with open(os.path.join(self.save_path, f'ddE_{model}_{size}.pkl'), 'wb') as f:
        #     pickle.dump(results, f)
        df_results = pd.DataFrame(results).T
        df_results.columns = ['r2', 'mae', 'rmse','rmse_t', 'secr']
        print(df_results)
        df_results.to_csv(os.path.join(save_path, f'ddE_{model}_{size}.csv'))
        
        return df_results
    


if __name__ == "__main__":
    metadata = pickle.load(open("./metadata/oc20_meta/oc20_data_metadata.pkl", "rb"))
    df_train = pd.read_pickle("./data/df_is2re_100k_new.pkl")
    df_val = pd.read_pickle("./data/df_is2re_val_25k_new.pkl")
    save_path = "/home/jovyan/CATBERT/results/dummy"
    
    models = ['catbert', 'cgcnn', 'dimenet', 'schnet', 'dimenetpp']
    print("============ Analysis Initiation ============")
    Results = SplitAnalysis(df_train, df_val, metadata, save_path)
    
    for model in models:
        print("============ " + model + " ============")
        print("------------ Splits ------------")
        Results.plot_val_splits(model, '100k')
        print("------------ Energy Difference ------------")
        Results.plot_energy_difference(model, '100k')

