# DLCM
The Double Layer Carbon Model (DLCM) is an advanced and comprehensive tool designed to simulate soil organic carbon (SOC) dynamics for both the top 20 cm layer (SOC20) and the deeper 20-100 cm layer (SOC20–100). The fundamental principle of the DLCM revolves around accurately predicting SOC dynamics by simulating the interactions between fresh organic matter inputs, microbial activity, and existing soil carbon pools. The DLCM defines four soil carbon pools, categorized based on their location within the soil profile and their decomposition rates. The model divides the soil profile into topsoil (0-20 cm) and subsoil (20–100 cm) layers to match the SOC maps of the corresponding two layers generated by data-driven models. Each of these layers contains a young carbon pool (CY) with a higher decomposition rate and an old carbon pool (CO) with a lower decomposition rate. Specifically, the topsoil contains the young topsoil carbon pool (CYt) and the old topsoil carbon pool (COt), while the subsoil contains the young subsoil carbon pool (CYs) and the old subsoil carbon pool (COs). These compartments help accurately simulate the dynamics of SOC by considering both the fast-cycling and slow-cycling components of organic matter decomposition. The DLCM optimizes initial SOC stocks using extensive spatial simulations, ensuring accurate baseline carbon levels. It also incorporates climate change responses, adjust decomposition rates based on climate and environmental changes, and lead to robust estimates under different climatic scenarios. The simulation process of the DLCM involves initializing SOC stocks with spatially detailed baseline data, adding organic matter inputs based on vegetation production, and simulating microbial decomposition while adjusting for climate variables such as temperature and soil moisture. The model performs layer-specific calculations to capture depth-specific SOC dynamics. It relies on comprehensive input data, including initial SOC stocks, climate data, and vegetation production to drive these simulations.
## What do I need to run DLCM?
* [![python](https://img.shields.io/badge/Python-3-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) and the required modules

The packages required to run DLCM are:

```
- numpy
- gdal
- netCDF4
- random
- numba
- pyswarm
- pandas
```
## How to run DLCM
We provide the source code of DLCM, and users can directly modify the input and output data paths in the Python 3 environment to run the model.
The operation of this model mainly includes three steps. The first step is to calibrate the model parameters. We drive the model based on the initial NPP data and the SOC maps of the surface 20cm and deep 20-100cm constructed based on machine learning algorithms in the 1980s. Using particle swarm optimization algorithm for parameter calibration. Due to the fact that our model simulates each pixel individually, a significant amount of computation is required to handle high spatial resolution data. We temporarily store the optimization parameters for each row of the raster image. After the parameter calibration of all pixels is completed, we will convert all optimized parameters into a raster map. The third step is to simulate the spatiotemporal dynamics of SOC based on the optimized parameter file. The spatiotemporal dynamic simulation requires the following input files.
## Step by step run the code
### Download the model input data
Download the model input data from https://doi.org/10.6084/m9.figshare.27646008.v2 (see Model Inputs section). This data includes NPP raster files, SOC raster files for different depths, and various parameter raster files.
### Modify the file paths in the source code
Open the source code and locate the if __name__ == '__main__': block. Update the file paths according to your local environment. Specifically:
- For the spatial parameter calibration part:
1. Set NPP_raster_file to the path of your NPP raster file (e.g., "model_input/NPP/Qinling_NPP_1982.tif").
2. Set SOC_20_raster_file and SOC_20_100_raster_file to the paths of your SOC raster files for the top 20 cm and 20 - 100 cm layers respectively.
3. Set fBNPP_chn, f20cm_chn, and f20_100cm_chn to the paths of your relevant parameter raster files.
4. Set opti_para_all_data_txt_filepath to the desired output path for the parameter optimization results in text format.
- For converting parameter optimization results to raster files:
1. Set opti_para_all_data_txt_filepath to the same path as used in the previous step.
2. Set reference_raster_filepath to the path of a reference raster file (e.g., the NPP raster file).
3. Set output_raster_path to the desired output path for the raster files.
- For the time - series simulation part:
1. Set NPP_time_series_filepath, Temp_series_filepath, SM_top_series_filepath, and SM_sub_series_filepath to the paths of your NPP, temperature, top - layer soil moisture, and sub - layer soil moisture time - series data folders respectively.
2. Set fBNPP_chn, f20cm_chn, and f20_100cm_chn to the paths of your relevant parameter raster files.
3. Set parameter_filepath to the path of your parameter files.
4. Set Output_SOC_series_filepath according to the simulation scenario you choose (K_driver_scenario).
### Choose the simulation scenario
Determine the value of K_driver_scenario according to your needs.
1. If K_driver_scenario == 1, the simulation will use only the fT and fW functions.
2. If K_driver_scenario == 2, the simulation will use only the fW function.
3. If K_driver_scenario is neither 1 nor 2, the simulation will use the combination of fT and fW functions.
### Run the code
After all the above steps are completed, run the Python script. The code will first perform spatial parameter calibration, then convert the parameter optimization results into raster files, and finally conduct time - series simulations based on your chosen scenario.
## Model Inputs
Data used to drive the DLCM include annual net primary production (NPP) from 1982 to 2018, SOC maps in the two soil layers during the 1980s, annual mean temperature maps from 1982 to 2018, and annual surface soil moisture and root-zone soil moisture maps from 1982 to 2018.
These data can be accessed at https://doi.org/10.6084/m9.figshare.27646008.v2.
