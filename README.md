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
## Input
By default, `analyze.py` expects that your data has been separated by channel into individual images according to this naming scheme:
```
<sample-ID>_fov<field-of-view-number>_ch<channel-number>.tif
```
While .tif is used for the example data and above, note that the images can be of any format readable by [scikit-image](https://scikit-image.org/docs/stable/api/skimage.io.html#). 

To modify the file pattern to suit your data, adjust the `regexpression` parameter in your configuration file. See the [wiki](https://github.com/panlilio/corrutil/wiki/Configuration-file-parameters) for details.

By default, results will be written to a timestamped 'results' folder in the current working directory. To specify a different output folder, adjust the `destdir` parameter in your configuration file: again, see [wiki](https://github.com/panlilio/corrutil/wiki/Configuration-file-parameters). 

## Outputs
When run as the main script, `analyze.py` will generate the following outputs:
* heatmaps of each correlational metric (.pdf)
* table of correlational metrics and intensity measurements (.csv)

A new, appended .csv is written after every dataset group is analyzed so that the analyses are not lost because of an interruption. When all groups are processed successfully, these intermediate .csvs are deleted: only the final, complete .csv is kept. In the future these files may be useful for restarting analyses at certain checkpoints, but this has not been implemented yet. If your analysis is interrupted, you will need to either redo the entire directory or keep the previous results and manually remove the image files that have already been analyzed.

See the [wiki](https://github.com/panlilio/corrutil/wiki/Understanding-the-results) for the mathematical definition of each correlation, and how to read the heatmaps and .csv headers. 
