# corrutil - Tools for correlational image analyses

Cross-channel correlational metrics included:
* Pearson correlation
* Spearman correlation
* Mutual information
* Overlap percentage based on threshold

Additional metrics:
* Intensity mean, median, minimum, maximum, standard deviation

## Getting started 
A minimal conda environment file is provided. The main dependencies are:
* numpy
* scipy
* skimage
* sklearn
* matplotlib

### Create conda environment
```
conda env create --file environment.yml
```

### Running the analyses
1. Activate the conda environment if needed
   ```
   conda activate corrutil
   ```
2. Navigate to the wherever you have downloaded the repo, e.g.
   ```
   cd ~/Downloads/corrutil
   ```
3. Run the analysis using default parameters on the dummy dataset provided, by providing the directory path
   ```
   python analyze.py data
   ```
   OR by using the JSON configuration file
   ```
   python analyze.py config.json
   ```
### Input format
By default, `analyze.py` expects that your data has been separated by channel into individual images according to this naming scheme:
```
<sample-ID>_fov<field-of-view-number>_ch<channel-number>.tif
```
While .tif is used for the example data and above, note that the images can be of any format readable by [scikit-image](https://scikit-image.org/docs/stable/api/skimage.io.html#). 

To modify the file pattern to suit your data, adjust the `regexpression` parameter in your configuration file. See [here](https://github.com/panlilio/corrutil/wiki/Configuration-file-parameters).
