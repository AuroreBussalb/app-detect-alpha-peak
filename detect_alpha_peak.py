#!/usr/local/bin/python3

import json
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def detect_alpha_peak(psd_welch, freqs):
    
    # Average across all channels
    psd_welch_mean = np.mean(psd_welch, axis=0)

    # Select band 7 - 14Hz #
    # Extract indices of alpha freqs
    indexes_alpha_freqs = [i for i, f in zip(range(0, len(freqs)), freqs) if f > 6.9  and f < 14.1]
    # Extract alpha freqs 
    alpha_freqs = np.take(freqs, indexes_alpha_freqs)
    # Extract psd in alpha freqs
    psd_in_alpha_freqs = np.take(psd_welch_mean, indexes_alpha_freqs)

    # Find peak #
    pic_loc = mne.preprocessing.peak_finder(psd_in_alpha_freqs)

    # Find the corresponding frequency #
    index_of_the_pic = int(pic_loc[0])
    alpha_freq_pic = alpha_freqs[index_of_the_pic]  # Apply the index of the pic in alpha_freqs, not in freqs!

    return alpha_freq_pic, psd_welch_mean 


# def plot_psd(psd_welch_mean, freqs, alpha_freq_pic):

#     plt.plot(freqs, psd_welch_mean)
   

def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Read the outputs of PSD app #

    # Load csv
    psd_welch_file = config.pop('psd_welch')

    # Extract PSD
    df_psd_welch = pd.read_csv(psd_welch_file)
    df_psd_welch = df_psd_welch.drop(["Unnamed: 0"], axis=1)
    psd_welch = df_psd_welch.to_numpy()

    # Extract freqs
    freqs = df_psd_welch.columns.to_numpy()
    freqs = freqs.astype(np.float)  

    # Detect alpha peak #
    alpha_freq_pic, psd_welch_mean = detect_alpha_peak(psd_welch, freqs)

    # Success message in product.json #
    success_message = f'Alpha peak successfully detected at {alpha_freq_pic:.2f}Hz.'   
    dict_json_product['brainlife'].append({'type': 'success', 'msg': success_message})

    # Save the dict_json_product in a json file #
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()

