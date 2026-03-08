# Folder & File Structure

##### confidential
- `PASSWORD.txt` --> Copernicus Data Space Ecosystem password.
- `SH_CLIENT_ID.txt` --> CDSE client id.
- `SH_CLIENT_SECRET.txt` --> CDSE client secret.
- `USERNAME.txt` --> Copernicus Data Space Ecosystem username.

##### data
- `dataset_ndwi/` --> folder with the NDWI rasters.
- `escales/` --> folder with the NDWI rasters for an additional reservoir, not included in the initial dataset. Additionally, it also includes the ground truth labels for Escales reservoir from 2020 to 2026.
- `terradets/` --> folder with the NDWI rasters for an additional reservoir, not included in the initial dataset. Additionally, it also includes the ground truth labels for Terradets reservoir from 2020 to 2026.
- `window_checks/` --> folder with the images to visually inspect the windows defined for each reservoir.
- `capacity_reservoirs.csv` --> ground truth labels for the 9 reservoirs included in the initial dataset.
- `dataset_ndwi.zip` --> zip file with the rasters in dataset_ndwi/.
- `escales_predictions.csv` --> predictions for the Escales reservoir for all three models; MSE, MAPE, and log(MSE).
- `master_labels_volume.csv` --> file that contains the ground truth volume that corresponds to each of the rasters.
- `riudecanyes_predictions.csv` --> predictions for the Riudecanyes reservoir for all three models; MSE, MAPE, and log(MSE).
- `siurana_predictions.csv` --> predictions for the Siurana reservoir for all three models; MSE, MAPE, and log(MSE).
- `terradets_predictions.csv` --> predictions for the Terradets reservoir for all three models; MSE, MAPE, and log(MSE).
- `test_labels.csv` --> test split of the master_labels_volume.csv file.
- `train_labels.csv` --> train split of the master_labels_volume.csv file.
- `val_labels.csv` --> validate split of the master_labels_volume.csv file.

##### models
- `dam_model_log_mse.pth` --> file with the model trained using log(MSE).
- `dam_model_mape.pth` --> file with the model trained using MAPE.
- `dam_model_mse.pth` --> file with the model trained using MSE.

##### narr
- `methodology.md` --> file with the reasoning behind the methodological decision taken.
- `structure` --> this file, which contains the folder structure and the file details.

##### output
- `figs/` --> folder with the relevant figures produced.

##### src
- `cnn.ipynb` --> file with the code to train, validate, and test a Convolutional Neural Network to predict the water volume in reservoirs from NDWI rasters.
- `get_labels.ipynb` --> file with the code that generates the file data/master_labels_volume.csv.
- `get_sentinel-2.ipynb` --> file with the code that computes the NDWI and downloads the rasters for each of the 9 dams across time.
- `get_terradets_sentinel-2.ipynb` --> file with the code that computes the NDWI and downloads the rasters for the Terradets reservoir.
- `inference.ipynb` --> file with the code that generates the predictions (using all three models: MSE, MAPE, log(MSE)) for Terradets reservoir from 2020 to 2026.