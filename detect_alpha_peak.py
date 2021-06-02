#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np
import os
import shutil
import pandas as pd


def detect_alpha_peak():
	pass


def main():
	    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)



    # Success message in product.json    
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Alpha peak successfully detected.'})

	# Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)

if __name__ == '__main__':
    main()

