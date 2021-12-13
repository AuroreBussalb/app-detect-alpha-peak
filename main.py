# Copyright (c) 2021 brainlife.io
##
# Author: Guiomar Niso
# Indiana University

# set up environment
import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)
    

# == GET CONFIG VALUES ==
fname = config['psd']
fmin  = config['fmin']
fmax  = config['fmax']

# == LOAD DATA ==
df_psd = pd.read_csv(fname, sep='\t')
canales = df_psd['channels'].copy()
#canales = df_psd.index.values

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
ifreqs = [i for i, f in zip(range(0, len(freqs)), freqs) if f > fmin  and f < fmax]
band_freqs = np.take(freqs, ifreqs)



# ==== FIND FREQUENCY  PEAK ====

channel_peak = []

# Prepare for Figure 1 containing all the channels
plt.figure(1)

if nchannels==1:
    fig, axs = plt.subplots(nchannels,1, figsize=(40, 20), facecolor='w', edgecolor='k')
else:
    fig, axs = plt.subplots(math.ceil(nchannels/10),10, figsize=(40, math.ceil(nchannels/10*2)), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace =.5, wspace=.2)
    axs = axs.ravel()

for channel in range(0, nchannels):
    
    # Find maxima with noise-tolerant fast peak-finding algorithm
    psd_channel = np.take(psd_welch[channel, :], ifreqs)
    pic_loc, pic_mag = mne.preprocessing.peak_finder(psd_channel, extrema=1, verbose=None)

    # From all the peaks found, get the main peak

    #If one peak found
    if pic_loc.size==1: 
        peak=pic_loc[0].copy()
        if peak==0: peak = math.nan # if it's the first value, then ignore, maybe just a decreasing curve

    #If no peak found
    if pic_loc.size==0: 
        peak=math.nan #(np.array([0]),np.array([0]))#NaN
        print('No peak found for channel: ',canales[channel])
    
    #If more than one peak found
    elif pic_loc.size>1:
        peak = np.where(psd_channel==max(pic_mag))[0][0] # take the max
        if peak==0: peak = pic_loc[np.argmax(pic_mag[1:,])+1] # if it's the first value, take the next max
        if peak==psd_channel.size-1: peak = pic_loc[np.argmax(pic_mag[0:-1])] # if it's the last value, take the next max

        print('Multiple peaks found for channel: ',canales[channel])

    #Get the frequency of the peak
    pic_freq = np.take(band_freqs,peak) if not math.isnan(peak) else math.nan
    channel_peak.append(pic_freq)
    
    # FIGURE 1
    axs[channel].plot(band_freqs,psd_channel)
    axs[channel].plot(np.take(band_freqs,pic_loc),pic_mag,'*')
    axs[channel].axvline(x=pic_freq,c='k',ls=':')
    axs[channel].set_title(canales[channel])
    axs[channel].set_xlim(fmin,fmax)

#Save Figure 1  
plt.savefig(os.path.join('out_figs','psd_allchannels.png'),dpi=20)
plt.close()

'''
# ==== FIND ALPHA MEAN VALUE ====
alpha_channel_peak = np.mean(psd_welch[:,ifreqs], axis=1)
'''

# Average value across all channels
mean_peak=np.nanmean(channel_peak, axis=0)


# == SAVE FILE ==
# Save to TSV file
df_alpha = pd.DataFrame(channel_peak, index=canales, columns=['peak'])
df_alpha.to_csv(os.path.join('out_dir','psd.tsv'), sep='\t')


# ==== PLOT FIGURES ====

# FIGURE 2
# Plot PSD
plt.figure(2)
plt.plot(freqs, psd_welch.transpose(), zorder=1) 
plt.xlim(xmin=0, xmax=max(freqs))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')

plt.axvline(x=mean_peak,c='k',ls=':')
# Save fig
plt.savefig(os.path.join('out_figs','psd_peak_frequency.png'))
plt.close()


# FIGURE 3
plt.figure(3)
#custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#sns.set_theme(style="ticks", rc=custom_params)
sns.set_theme(style="ticks")
sns.histplot(data=channel_peak, binwidth=0.25,kde=True,kde_kws={'cut':10})
#plt.xlim(xmin=fmin, xmax=fmax)
plt.xlabel('Peak frequency (Hz)')
sns.despine()
# Save fig
plt.savefig(os.path.join('out_figs','hist_peak_frequency.png'))
plt.close()
