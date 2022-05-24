# PurpleAirSmoke
This repository contains the code and python environments used in the analysis of a paper under review and Environmental Research: Health. If you would like to use these codes or find any errors in them, please contact me at the email listed on my github profile. 

To perform the analysis presented in the paper, in prep for GeoHealth, the following scripts are run in the following order:

**Stage 1: Create lists of co-located indoor and outdoor monitors in the PurpleAir data. These codes were run on my local machine and use the “local” python3 environment in this repository**.

Cross_check_sensor_lists.py

Loads site lists created by Bonne Ford and Jude Bayham, files purpleair_sitelist_global_4.csv and Jude_sensor_metadata.csv contact me for these files if interested. More updated versions can be downloaded via the purple air webpage. These files contain a list of public PurpleAir monitors, thingspeak ids, monitor type, indoor/outdoor flag, and latitude and longitude for the monitors. Use local python packages.

Sensor_list_sites.py

Loads moved IDs output from cross_check_sensor_lists.py, Bonne site list, and a county-level shapefile. Use local python packages.

match_monitor_wSVI.py

Loads co-located indoor and outdoor sensor list output from sensor_list_sites.py and census tract level SVI data from the CDC/ATSDR which can be accessed at https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html Use local python packages.Makes Figure 2f.

**Stage 2: Clean and process the PurpleAir monitor data. These codes use the "remote" python environment in this repository.**

process_PA_raw.py

Loads western US sensor list output from sensor_list_sites.py. Loads PurpleAir raw sensor files downloaded by Bonne Ford. Processes the raw sensor data, checks for missing files (due to no monitor data in 2020), and makes merges of the A and B sensor data for each monitor, where available (only certain monitor types have two sensors). Use remote python packages.

AB_clean_stats.py

Loads processed A-B merges output from process_PA_raw.py. Loads the processed metadata with the processed file names. Cleans data for values outside plantower effective range, checks for A-B agreement, applies correction factors, and creates an indoor-outdoor merge. Use remote python packages.

add_smoke2PA.py

Loads cleaned, in-out merges created in AB_clean_stats.py. Loads processed file metadata from process_PA_raw.py. Loads shapefiles of smoke plumes from NOAA’s Hazard Mapping System (HMS) fire and smoke product. Available online here: https://www.ospo.noaa.gov/Products/land/hms.html  Assigns a smoke-day flag to the merged indoor-outdoor observations for each file and calculates a smoke event length (which we ultimately don’t use in the analysis). Use remote python packages.

Average_processed_data.py

Loads cleaned, inout merges with HMS flags created in add_smoke2PA.py, and processed metadata file. Averages data to daily and hourly averages, creates smoke day flag (using the PM and HMS criteria), and saves averaged merges. Use remote python packages.

**Stage 3: Perform final analysis on cleaned monitor data and make figures.**

Initial_analysis_allPA.py

Loads daily-averaged, in-out merges created in average_processed_data.py, clean metadata file, processed metadata file, and original metadata file
Re-run for each different region, select region in the code header. Makes the components of Figure 5, which are then stitched together in Powerpoint.
Makes Figures S2-S6, hourly (need to select hourly in header for these figures). Use remote python packages.

Boxplot_all_counties.py

Loads daily-averaged, in-out merges created in average_processed_data.py, clean metadata file, and processed metadata file. Makes Figures 3 and S2 (run separately, select SI or main version in header). Use remote python packages.

Calc_inout_ratio.py

Loads daily-averaged, in-out merges created in average_processed_data.py, original metadata file, and processed metadata file. Calculates average values and statistics for each western US monitor and saves to file for plotting on local machine. Run for main and SI versions. Use remote python packages.

plot_PAratio_stats.py

Loads monitor ratio data from calc_inout_ratioy.py, sensor list with SES data from match_monitor_wSVI.py and data on cleaning stats output from AB_clean_stats.py
Run with main and SI versions of data output from calc_inout_ratio.py Use local python packages. Makes Figures 1, 4, S1, S3, S4 and S10-S13. Note for figures S4 and S10-S13, the "main" version should be used even though these are SI figures. 
