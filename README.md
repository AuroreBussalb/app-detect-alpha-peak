# app-peak-frequency

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.469-blue.svg)](https://doi.org/10.25663/brainlife.app.469)

Detect individual alpha peak in MEG signals.

## app-peak-frequency documentation

1) Detect peak frequency for each channel for a guiven bandwith
2) Input file is:
    * a `.csv` file containing the Power Spectral Densities
5) Ouput files are:
    * a `.csv` file containing the alpha peak frequency for each channel and the mean over all channels,
    * plot of the mean Power Spectral density over all channels,
    * plot of the mean Power Spectral density for each channel.

### Authors
- [Guiomar Niso](guiomar.niso@ctb.upm.es)

### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your code and publications. Copy and past the following lines into your repository when using this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

### Citations
1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Running the App 

### On Brainlife.io

This App is still private on Brainlife.io.

### Running Locally (on your machine)

1. git clone this repo
2. Inside the cloned directory, create `config.json` with the same keys as in `config.json.example` but with paths to your input 
   files and values of the input parameters. For instance:

```json
{
  "fif": "rest1-raw.fif"
}
```

3. Launch the App by executing `main`

```bash
./main
```

## Output

The output files are a `.csv` file and two plots in `.png`.


