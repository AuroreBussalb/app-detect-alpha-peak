#!/usr/local/bin/python3

import json
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil


def get_alpha_freqs(freqs):

    # Select band 7 - 14Hz #

    # Extract indices of alpha freqs
    indexes_alpha_freqs = [i for i, f in zip(range(0, len(freqs)), freqs) if f > 6.9  and f < 14.1]
    # Extract alpha freqs 
    alpha_freqs = np.take(freqs, indexes_alpha_freqs)

    return alpha_freqs, indexes_alpha_freqs 


def detect_alpha_peak_mean(psd_welch, alpha_freqs, indexes_alpha_freqs):


    # Average PSD across all channels
    psd_welch_mean = np.mean(psd_welch, axis=0)    

    # Get the std of the mean
    psd_welch_std = np.std(psd_welch, axis=0) 

    # Extract psd in alpha freqs
    psd_in_alpha_freqs_mean = np.take(psd_welch_mean, indexes_alpha_freqs)

    # Find peak 
    pic_loc = mne.preprocessing.peak_finder(psd_in_alpha_freqs_mean)

    # Find the corresponding frequency 
    index_of_the_pic = int(pic_loc[0])
    alpha_freq_pic_mean = alpha_freqs[index_of_the_pic]  # Apply the index of the pic in alpha_freqs, not in freqs!

    return alpha_freq_pic_mean, psd_welch_mean, psd_welch_std, psd_in_alpha_freqs_mean 


def detect_alpha_peak_per_channels(psd_welch, alpha_freqs, indexes_alpha_freqs):

    # Get alpha peak frequency for all channels # 

    alpha_freq_pic_per_channel = []
    for channel in range(0, psd_welch.shape[0]): 
        # Extract psd in alpha freqs
        psd_in_alpha_freqs_per_channel = np.take(psd_welch[channel, :], indexes_alpha_freqs)
        # Find peak 
        pic_loc = mne.preprocessing.peak_finder(psd_in_alpha_freqs_per_channel)
        pic_loc = pic_loc[0]
        if len(pic_loc) > 1:  # if more than 1 peak is found
            pic_loc = max(pic_loc)
        # Find the corresponding frequency
        index_of_the_pic = int(pic_loc)
        alpha_freq_pic = alpha_freqs[index_of_the_pic]  # Apply the index of the pic in alpha_freqs, not in freqs!
        alpha_freq_pic_per_channel.append(alpha_freq_pic)

 
    return alpha_freq_pic_per_channel, psd_in_alpha_freqs_per_channel 


def plot_psd_mean(freqs, alpha_freq_pic_mean, psd_welch_mean, psd_welch_std, alpha_freqs, psd_in_alpha_freqs_mean):

    plt.figure()
 
    # Get the index of alpha peak
    id_alpha_peak =  np.where(freqs==alpha_freq_pic_mean)

    # Define lim
    plt.xlim(xmin=0, xmax=max(freqs))

    # Plot spectrum
    plt.plot(freqs, psd_welch_mean, zorder=1)   

    # Plot a red point on the alpha peak
    plt.scatter(alpha_freq_pic_mean, psd_welch_mean[id_alpha_peak], marker='o', color="red", label='alpha peak frequency', zorder=3)

    # Plot std of the mean as shaded area
    plt.fill_between(freqs, psd_welch_mean-psd_welch_std, psd_welch_mean+psd_welch_std, alpha=0.5, label='standard deviation', zorder=2)

    # Shadow the frequencies in which we look for the peak
    # plt.ylim(ymin=min(psd_welch_mean))
    plt.axvline(x=min(alpha_freqs), zorder=4, color='r', linestyle='--', alpha=0.2, label='alpha band')
    plt.axvline(x=max(alpha_freqs), zorder=4, color='r', linestyle='--', alpha=0.2)

    # Define labels
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Mean power spectrum across all channels')
    plt.legend()

    # Save fig
    plt.savefig('out_dir/psd_mean.png')


def plot_psd_per_channels(freqs, alpha_freq_pic_per_channel, psd_in_alpha_freqs_per_channel, psd_welch, alpha_freqs):

    plt.figure()
 
    # Get the index of each alpha peak
    # list_alpha_peak_id = []
    # for freq_channels in alpha_freq_pic_per_channel:
    #     id_alpha_peak = np.where(freqs==freq_channels)
    #     list_alpha_peak_id.append(id_alpha_peak)

    # Plot spectrum
    # for channel, id_papf in zip(range(0, len(alpha_freq_pic_per_channel)), list_alpha_peak_id):
    #     plt.plot(freqs, psd_welch[channel], zorder=1)  
          # plt.scatter(alpha_freq_pic_per_channel[channel], psd_welch[channel, id_papf], marker='o', color="red", zorder=3, alpha=0.2, size=0.2)
    for channel in range(0, len(alpha_freq_pic_per_channel)):
        plt.plot(freqs, psd_welch[channel], zorder=1)  

    # Define lim
    plt.xlim(xmin=0, xmax=max(freqs))
 
    # Plot alpha band
    plt.axvline(x=min(alpha_freqs), zorder=4, color='r', linestyle='--', alpha=0.6, label='alpha band')
    plt.axvline(x=max(alpha_freqs), zorder=4, color='r', linestyle='--', alpha=0.6)

    # Define labels
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power spectrum for all channels')
    plt.legend()

    # Save fig
    plt.savefig('out_dir/psd_channels.png')


def main():

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Load csv
    path_to_input_file = config.pop('psd')
    # To be able to read input raw datatype
    # if "out_dir/." in path_to_input_file:
    #     path_to_input_file = path_to_input_file.replace('out_dir/.', 'out_dir/psd.csv')


    # Read the outputs of PSD app #

    # Extract PSD
    df_psd_welch = pd.read_csv(path_to_input_file)
    df_psd_welch = df_psd_welch.drop(["Unnamed: 0"], axis=1)
    psd_welch = df_psd_welch.to_numpy()

    # Extract freqs
    freqs = df_psd_welch.columns.to_numpy()
    freqs = freqs.astype(np.float)  

    # Get alpha freqs
    alpha_freqs, indexes_alpha_freqs = get_alpha_freqs(freqs)

    # Detect alpha peak on average channels #
    alpha_freq_pic_mean, psd_welch_mean, psd_welch_std, psd_in_alpha_freqs_mean = detect_alpha_peak_mean(psd_welch, alpha_freqs, indexes_alpha_freqs)

    # Detect alpha peak per channels #
    alpha_freq_pic_per_channel, psd_in_alpha_freqs_per_channel = detect_alpha_peak_per_channels(psd_welch, alpha_freqs, indexes_alpha_freqs)


    # Create a DataFrame with alpha peak values #

    # Values for each channel
    channels = [f"channel_{i}" for i in range(0, len(alpha_freq_pic_per_channel))]
    d_all_channels = {'channels': channels, 'alpha peak frequency': alpha_freq_pic_per_channel} 
    df_alpha_peaks = pd.DataFrame(data=d_all_channels)

    # Value for mean PSD across channels
    d_mean_channels = {'channels': "mean channels", 'alpha peak frequency': alpha_freq_pic_mean}
    df_alpha_peaks = df_alpha_peaks.append(d_mean_channels, ignore_index=True)

    # Save it into a csv
    df_alpha_peaks.to_csv('out_dir/alpha_peak_frequency.csv', index=False)


    # Plot spectrum #

    # Mean spectrum
    plot_psd_mean(freqs, alpha_freq_pic_mean, psd_welch_mean, psd_welch_std, alpha_freqs, psd_in_alpha_freqs_mean)

    # All channels
    plot_psd_per_channels(freqs, alpha_freq_pic_per_channel, psd_in_alpha_freqs_per_channel, psd_welch, alpha_freqs)


if __name__ == '__main__':
    main()

