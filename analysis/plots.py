import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
import numpy as np
import seaborn as sns
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)


def parity_plot(label, pred, 
                plot_type, 
                xlabel='$\Delta E_{ads}$ [eV]', 
                ylabel='$\Delta \hat{E}_{ads}$ [eV]', 
                margin=False, color=None,
                xylim= [-20, 20]):
    '''
    targets_pred: x value
    targets_val: y value 
    residuals: targets_pred - targets_val 
    plot_type: hexabin / scatter
    xlabel: x-label name
    ylabel: y-label name
    delta: unit dE if True, unit E if False
    color: customize the color
    '''
    # Plot
    residuals = pred - label
    lims = xylim
    if plot_type == 'hexabin':
        grid = sns.jointplot(x=label, 
                             y=pred,
                             kind='hex',
                             bins='log',
                             extent=lims+lims, 
                             color=color)
        
    elif plot_type == 'scatter':
        grid = sns.jointplot(x=label, 
                             y=pred, 
                             kind='scatter',
                             color=color)
        
    ax = grid.ax_joint
    _ = ax.set_xlim(lims)
    _ = ax.set_ylim(lims)
    _ = ax.plot(lims, lims, '--', c='grey')
    _ = ax.set_xlabel(f'{xlabel}', fontsize=16)
    _ = ax.set_ylabel(f'{ylabel}', fontsize=16)
    
    # Calculate the error metrics
    mae = mean_absolute_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    r2 = r2_score(label, pred)
    
    # Report
    text = ('\n' +
            '  $R^2$ = %.2f\n' % r2 +
            '  MAE = %.2f eV\n' % mae + 
            '  RMSE = %.2f eV\n' % rmse
            )
    
    _ = ax.text(x=lims[0], y=lims[1], s=text,
                horizontalalignment='left',
                verticalalignment='top', fontsize=12)
    if margin == False:
        grid.ax_marg_x.remove()
        grid.ax_marg_y.remove()
        plt.axvline(xylim[1], c='k', lw=2.2)
        plt.axhline(xylim[1], c='k', lw=1.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return r2, mae, rmse


def get_array_for_grouping(df, metadata, ml_preds):
    '''
    df: dataframe containing texts and target dft energies
    metadata: oc20 metadata dictionary
    ml_preds: dictionary of multiple gnn predictions
    '''
    code = df['id'].to_numpy()
    dft = df['target'].to_numpy()
    ml = np.array([ml_preds[id] for id in code])
    ads_id = np.zeros(len(code))
    bulk_id = np.zeros(len(code))
    for i in range(len(code)):
        ads_id[i] = metadata[code[i]]["ads_id"]
        bulk_id[i] = metadata[code[i]]["bulk_id"]
    return code, dft, ml, ads_id, bulk_id


def grouping_fast(code, dft_energy, ml_energy, ads, bulk):
    '''
    code: code index of adslab
    dft: dft-calculated energy 
    ml: ml-predicted ml energy 
    ads: oc20 ads index
    bulk: oc20 bulk index
    '''
    cat_swap, ads_swap, conf_swap, all_swap = {}, {}, {}, {}
    total_num = 0

    for i, code_i in enumerate(code):
        for j, code_j in enumerate(code[i+1:], start=i+1):
            if ads[i] == ads[j] and bulk[i] == bulk[j]:
                cat_swap[(code_i, code_j)] = (dft_energy[i] - dft_energy[j], ml_energy[i] - ml_energy[j])
            elif ads[i] != ads[j] and bulk[i] == bulk[j]:
                ads_swap[(code_i, code_j)] = (dft_energy[i] - dft_energy[j], ml_energy[i] - ml_energy[j])
            elif ads[i] == ads[j] and bulk[i] != bulk[j]:
                conf_swap[(code_i, code_j)] = (dft_energy[i] - dft_energy[j], ml_energy[i] - ml_energy[j])
            elif ads[i] != ads[j] and bulk[i] != bulk[j]:
                all_swap[(code_i, code_j)] = (dft_energy[i] - dft_energy[j], ml_energy[i] - ml_energy[j])
            total_num += 1

    print('total combinations: ', total_num)
    print(f'cat_swap ratio: {round(100 * len(cat_swap) / total_num,2)} %')
    print(f'ads_swap ratio: {round(100 * len(ads_swap) / total_num,2)} %')
    print(f'conf_swap ratio: {round(100 * len(conf_swap) / total_num,2)} %')
    print(f'similar pair ratio: {round(100 * (len(cat_swap) + len(ads_swap) + len(conf_swap)) / total_num,2)} %')


    return cat_swap, ads_swap, conf_swap, all_swap

    