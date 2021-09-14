# Copyright (c) 2020 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#
# Author: Guiomar Niso
# Indiana University

# Required libraries
# pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks fire

# set up environment
#import mne-study-template
import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)
    

# == GET CONFIG VALUES ==
fname = config['psd']

# == LOAD DATA ==
df_psd = pd.read_csv(fname)
canales = df_psd['channels'].copy()

#Number of frequencies computed for the PSD
nfreqs = df_psd.shape[1]
df = df_psd.iloc[:, 1:nfreqs].copy() # To avoid the case where changing df also changes df_psd
#List of frequencies
freqs = df.columns.to_numpy()
freqs = freqs.astype(float)
#PSD values
psd_welch = df.to_numpy()
#Number of channels
nchannels = psd_welch.shape[0]


# Extract the frequencies that fall inside the alpha band
fmin=6.9
fmax=14.1
ifreqs = [i for i, f in zip(range(0, len(freqs)), freqs) if f > fmin  and f < fmax]
alpha_freqs = np.take(freqs, ifreqs)


# ==== FIND ALPHA PEAK ====

alpha_channel_peak = []

# Prepare for Figure 1 containing all the channels
plt.figure(1)
fig, axs = plt.subplots(math.ceil(nchannels/10),10, figsize=(10, math.ceil(nchannels/10*0.5)), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
axs = axs.ravel()


for channel in range(0, nchannels):
    
    # Find maxima with noise-tolerant fast peak-finding algorithm
    psd_channel = np.take(psd_welch[channel, :], ifreqs)
    pic_loc, pic_mag = mne.preprocessing.peak_finder(psd_channel, extrema=1, verbose=None);

    # From all the peaks found, get the main peak

    #If one peak found
    if pic_loc.size==1: 
        peak=pic_loc[0].copy()

    #If no peak found
    if pic_loc.size==0: 
        peak=(np.array([0]),np.array([0]))
        print('No peak found for channel: ',canales[channel])
    
    #If more than one peak found
    elif pic_loc.size>1:
        pic_mag_list=pic_mag.tolist()    
        peak = pic_mag_list.index(max(pic_mag_list)) # max? average? (GUIO)
        print('Multiple peaks found for channel: ',canales[channel])

    #Get the frequency of the peak
    pic_freq = np.take(alpha_freqs,peak)
    alpha_channel_peak.append(pic_freq)
    
    #Plot Figure 1
    axs[channel].plot(alpha_freqs,psd_channel);
    axs[channel].plot(np.take(alpha_freqs,pic_loc),pic_mag,'*');
    axs[channel].axvline(x=pic_freq,c='k',ls=':');
    #axs[channel].set_title(canales[channel])  

#Save Figure 1  
plt.savefig(os.path.join('out_dir2','psd_allchannels.png'),dpi=10)

# Average of the peak of all the channels
mean_alpha_peak=np.mean(alpha_channel_peak, axis=0)


# == SAVE FILE ==
# Save to CSV file (could be also TSV)
df_alpha = pd.DataFrame(alpha_channel_peak, index=canales, columns=['alpha_peak'])
df_psd.to_csv(os.path.join('out_dir','alpha_peak.csv'))


# Read CSV file
#df = pd.read_csv("df_psd.csv")
#print(df)


# ==== PLOT FIGURES ====

# FIGURE 2
# Plot PSD

plt.figure(2)
plt.plot(freqs, psd_welch.transpose(), zorder=1) 
plt.xlim(xmin=0, xmax=max(freqs))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
#plt.title('PSD alpha peak: ',mean_alpha_peak)  

# SHADE ALPHA BAND (GUIO)

plt.axvline(x=mean_alpha_peak,c='k',ls=':');
# Save fig
df_psd.to_csv(os.path.join('out_dir2','psd_alpha_peak.png'),dpi=10)

