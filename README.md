# Grineski_et_al_dust_projections
This repository contains output files and code relevant for the Great Salt Lake Dust projections. This code originally written by Derek Mallia and is based on the CMAQ implementation of FENGSHA.<br><br>

The python scripts compute dust emissions using an algorithem develop by Daniel Tong (FENGSHA). This uses a updated version presented in Mallia et al. 2017, which includes updated friction velocity thresholds 
for Playa surfaces. This code differs from the Mallia et al. approach as it uses HRRR model output instead of WRF.<br><br>

Input files include:<br>
(1) HRRR model data, which can be downloaded from the NOAA's AWS archive: https://aws.amazon.com/marketplace/pp/prodview-yd5ydptv3vuz2<br>
(2) Soil and landuse data, in this case, processed by WRF's WPS: https://home.chpc.utah.edu/~u0703457/Grineski_et_al_2024/model_inputs/<br>
(3) GSL Bathymetry data from the USGS: https://home.chpc.utah.edu/~u0703457/Grineski_et_al_2024/model_inputs/<br>
(4) Soil fraction data from NASA: https://ldas.gsfc.nasa.gov/gldas/soils<br><br>

Note that these files can be modified to fit the user needs, but might require edits to the Python code by the user. The code was written by DVM on 7/28/2022. For any questions about this data, 
contact Derek at Derek.Mallia@utah.edu <br><br>

---------
Mallia, D. V., A. Kochanski, C. Pennell, W. Oswald, and J. C. Lin (2017), Wind-blown dust modeling using a 
backward Lagrangian particle dispersion model. J. Appl. Meteor. Climate., 56, 2845–2867.<br>


