#!/usr/local/bin/python3

import json
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil


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

    return alpha_freq_pic, psd_welch_mean, alpha_freqs, psd_in_alpha_freqs  


def plot_psd(freqs, alpha_freq_pic, psd_welch_mean, alpha_freqs, psd_in_alpha_freqs):
 
    # Get the index of alpha peak
    id_alpha_peak =  np.where(freqs==alpha_freq_pic)

    # Plot spectrum
    plt.plot(freqs, psd_welch_mean)   

    # Plot a red point on the alpha peak
    plt.plot(alpha_freq_pic, psd_welch_mean[id_alpha_peak], marker='o', markersize=3, color="red")

    # Shadow the frequencies in which we look for the peak
    plt.ylim(ymin=min(psd_welch_mean))
    plt.fill_between(alpha_freqs, psd_in_alpha_freqs, min(psd_welch_mean), alpha=0.2)

    # Define labels
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')

    # Save fig
    plt.savefig('psd.png')


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Load csv
    path_to_input_file = config.pop('psd_welch')
    # To be able to read input raw datatype
    if "out_dir/." in path_to_input_file:
    	path_to_input_file = path_to_input_file.replace('out_dir/.', 'out_dir/psd.csv')

    # Read the outputs of PSD app #

    # Extract PSD
    df_psd_welch = pd.read_csv(path_to_input_file)
    df_psd_welch = df_psd_welch.drop(["Unnamed: 0"], axis=1)
    psd_welch = df_psd_welch.to_numpy()

    # Extract freqs
    freqs = df_psd_welch.columns.to_numpy()
    freqs = freqs.astype(np.float)  

    # Detect alpha peak #
    alpha_freq_pic, psd_welch_mean, alpha_freqs, psd_in_alpha_freqs = detect_alpha_peak(psd_welch, freqs)

    # Plot spectrum #
    plot_psd(freqs, alpha_freq_pic, psd_welch_mean, alpha_freqs, psd_in_alpha_freqs)

    # Success message in product.json #
    success_message = f'Alpha peak successfully detected at {alpha_freq_pic:.2f}Hz.'   
    dict_json_product['brainlife'].append({'type': 'success', 'msg': success_message})

    # Save the dict_json_product in a json file #
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()

