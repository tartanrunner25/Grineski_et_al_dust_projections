#This script computes dust emissions using an algorithem develop by Daniel Tong (FENGSHA). This uses a
#updated version presented in Mallia et al. 2017, which includes updated friction velocity thresholds 
#for Playa surfaces. This code differs from the Mallia et al. approach as it uses HRRR model output 
#instead of WRF.
#
#Mallia, D. V., A. Kochanski, C. Pennell, W. Oswald, and J. C. Lin (2017), Wind-blown dust modeling using a 
#backward Lagrangian particle dispersion model. J. Appl. Meteor. Climate., 56, 2845–2867.
#
#Written by DVM 7/28/2022

import numpy as np
import xarray as xr
import pandas as pd
import os.path as osp
import re, glob, sys, os
import re
import netCDF4 as nc
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from FENGSHA_dust_emiss_funcs import regrid_soilT, regrid_landT, fricT_table, landU_table
from FENGSHA_dust_emiss_funcs import assign_fricT, find_grib, met_regrid, regrid_soilF, compute_erode
from FENGSHA_dust_emiss_funcs import adjust_gsl

#Define our constants used to compute dust emissions
Rd = 287.058        #Dry gas constant
g = 9.81            #Gravity
A = 40              #FENGSHA scale factor (see Atmos. Chem. Phys. Dong et al. 2015)
gmad = 0.78         #Factor to convert geometric to aerodynamic diameter (see CMAQ code)
pm_ratio = 0.15     #Ratio between PM2.5 and PM10

#Input paths. Includes paths for HRRR data (hrrr_path), which provides friction velocities, soil type and landuse
#classificaions (soilT_path), and soil % fractions (soilF_path). Soil type data simply from WRF WPS geogrid output
#file since I am lazy and this grids up everything nicely for me. GSL bathymetry data also included if we want to
#shrink the GSL to current lake levels (or do projections)
hrrr_path = "./hrrr_data/hrrr/"
soilT_file = "../data/USGS_soil_type/geo_em.d01.nc"
soilF_file = "../data/GLDAS_soil/GLDASp4_soilfraction_025d.nc4"
gsl_file = '../../GSL_bathymetry/GSL_bathymetry_new.nc'

#Output path 
out_path = './dust_emissions_out/test/'

#Domain extent of emissions and the grid spacing in decimal degrees (lat/lon). Order of domain extent is
#lon min, lon max, lat min, and lat max
grid_resolution = .05
domain_extent = [-125.,-102,31,48]            #Default
#domain_extent = [-125.025,-102,31.025,48]    #For Jeff  

#Times we want to run dust emission model over? Either specify a time range or read in a file
#with times lised by row (single column) in the format as: YYYY-MM-DD HH:00:00
#emiss_times1 = pd.date_range(start='2022-03-14', end = '2022-03-17', freq='H')
#emiss_times2 = pd.date_range(start='2022-03-28', end = '2022-04-01', freq='H')
#emiss_times3 = pd.date_range(start='2022-04-04', end = '2022-04-06', freq='H')
#emiss_times4 = pd.date_range(start='2022-04-18', end = '2022-05-09', freq='H')
#emiss_times5 = pd.date_range(start='2022-05-18', end = '2022-05-21', freq='H')
#emiss_times = np.concatenate([emiss_times1,emiss_times2,emiss_times3,emiss_times4,emiss_times5])
emiss_times1 = pd.date_range(start='2023-03-27', end = '2023-03-31', freq='H')
emiss_times2 = pd.date_range(start='2023-04-10', end = '2023-04-14', freq='H')
emiss_times3 = pd.date_range(start='2023-04-16', end = '2023-04-20', freq='H')
emiss_times = np.concatenate([emiss_times1,emiss_times2,emiss_times3])
#my_times = pd.read_csv('UT09_emission_times_edited.csv',header=None)
#emiss_times = pd.to_datetime(my_times[0])

#What level do we want to set the GSL to? If we don't want to adjust the GSL, set the GSL_adjust
#flag to false. If GSL_adjust is set to true, lake_level must be set to some value!
GSL_adjust = True
lake_level = 1277

#Need to get our static data (soil & landuse type, and soil fraction) on a common grid.
#Lets define this common grid going forward as a mesh grid
xi = np.arange(domain_extent[0],domain_extent[1],grid_resolution)
yi = np.arange(domain_extent[2],domain_extent[3],grid_resolution)
xi,yi = np.meshgrid(xi,yi)

#Regrid soil type data
soil_cat = regrid_soilT(soilT_file,xi,yi)

#Regrid landuse type
land_cat = regrid_landT(soilT_file,xi,yi)

#Regrid soil fraction data for silt, sand, and clay
siltF, sandF, clayF = regrid_soilF(soilF_file,xi,yi)

#Retrieve table for friction velocity thresholds, assigned by soil type. Do the same 
#for the landuse type.
frict_table = fricT_table()
landuse_table = landU_table()

#Do we want to adjust the GSL lake level? If so, lets correct our soil types around the GSL.
if GSL_adjust == True:
    land_cat,soil_cat = adjust_gsl(gsl_file,lake_level,land_cat,soil_cat,xi,yi,grid_resolution)

#Assign friction velocity threshold values based on soil type and landuse category
fricT = assign_fricT(frict_table,landuse_table,land_cat,soil_cat)

#Compute erodibility based on FENSGA soil use type. First, convert land_cat based on USGS
#data and convert it into the 4 landuse types recognized by FENGHSA: Shrubland [1], 
#Shrubgrass [2], barren [3], and none of the above [-1]
erode = compute_erode(land_cat,landuse_table)

#We have all of the static data that we need, lets loop through each emission time that we want 
#to process, read in our grib data if its available, and compute our emissions.
for t in range(0, len(emiss_times)):
    
    print(str(emiss_times[t]))

    #Check to see if our file exists, if not, skip to next iteration
    file_check, grib_file = find_grib(hrrr_path,emiss_times[t])

    if file_check == False:
        print('No grib file found, jump to next time iteration')
        continue

    #Lets open our grib file and read in the varibles we want.
    grib_data = xr.open_dataset(grib_file, engine='cfgrib')
    lon2D = np.array(grib_data.coords['longitude']-360)
    lat2D = np.array(grib_data.coords['latitude'])
    fricv2D = np.array(grib_data['fricv'].values)
    temp2D = np.array(grib_data['t2m'].values)
    press2D = np.array(grib_data['sp'].values)

    #Lets regrid our meteorological variables to the same grid of our static data
    fricV = met_regrid(fricv2D, lon2D, lat2D, xi, yi)
    temp = met_regrid(temp2D, lon2D, lat2D, xi, yi)
    press = met_regrid(press2D, lon2D, lat2D, xi, yi)

    #Convert pressure and temperature to air density using ideal gas law (ro = P/RT) 
    #leaving us with units of kg/m3. Convert to g/m3 by multiplying by 1000.
    air_density = press/(Rd*temp)*1000

    #Identify areas where the friction velocity exceeds the friction velocity threshold. Used
    #as a diagnoistic variable.
    dust_emitter = fricV - fricT
    dust_emitter = np.nan_to_num(dust_emitter,0)
    dust_emitter[dust_emitter<0] = 0

    ########################################################################################
    #### Dust emission formula from Fu et al. 2014; Dong et al. 2015:                   ####
    ####                                                                                ####
	####        F   =  K x A x (ro/g) x SEP x U* x ( U*^2 - Ut*^2 ) x Si                ####
    ####                                                                                ####
	####        F   = emission flux                                  [g/m**2-s]         ####
	####        K   = vertical flux to horizontal sediment ratio     [1/m]              ####
	####        A   = 0~3.5  mean = 2.8                              [unitless]         ####
	####        r/g = ratio of air density to gravity=120.8          [g-s**2/m**4]      ####
	####        U*  = friction velocity                              [m/s]              ####
	####        Ut* = threshold friction velocity                    [m/s]              ####
	####        SEP = Soil erodible potential			             [unitless]         ####
    ####        Si  = Gridcell erodibility fraction                  [unitless]         ####
    ########################################################################################

    #Compute soil erodibility factor (SEP)
    SEP = (.08*clayF) + (1*siltF) + (.12*sandF)

    #Compute the the ratio of vertical to horizontal flux, which is a function of the clay
    #fraction of the soil. Initialize a grid for K before computing. Note, papers say % but
    #this is a fraction in the code so no adjustments needed.
    K = clayF * 0
    K[clayF <= .2] = 10.0**((13.4 * clayF[clayF <= .2]) - 6.0)
    K[clayF > .2] = .0002

    #Ok, at this point we have everything we need compute dust emissions
    dust_flux = K*A*(air_density/g)*SEP*fricV*(fricV**2 - fricT**2)*erode

    #Can't have a negative flux of dust so set those to 0 since it makes that we didn't 
    #have strng enough winds to loft dust. Along those lines set nan values to 0 (likely 
    #values at the boundaries that are nan from interpolation)
    dust_flux[dust_flux<=0]=0
    dust_flux[np.isnan(dust_flux)] = 0

    #Estimate our PM2.5 flux fraction
    pm25_flux = dust_flux*pm_ratio
    pm10_flux = dust_flux*(1-pm_ratio)

    #Lets save our model run to a netcdf file. First, define the file name for 
    #our netcdf file
    etime = str(emiss_times[t])
    nc_date = etime[0:10]
    nc_hour = etime[11:13]
    nc_prefix = str(nc_date+'_'+nc_hour)
    nc_filename = str(out_path+nc_prefix+'_dust_emissions.nc')

    #Create a blank netcdf file with the name that we prescribed above
    ds = nc.Dataset(nc_filename, 'w', format='NETCDF4')

    #Define our lat/lons
    lat = ds.createDimension('lat', len(np.unique(yi)))
    lon = ds.createDimension('lon', len(np.unique(xi)))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))

    #What variables do we want to add to our netcdf file?
    pm25_nc = ds.createVariable('Dust_emission (PM2.5)', 'f4', ('lat', 'lon',))
    pm25_nc.units = 'g/m2 s'
    pm10_nc = ds.createVariable('Dust_emission (PM10)', 'f4', ('lat', 'lon',))
    pm10_nc.units = 'g/m2 s'
    fricV_nc = ds.createVariable('Friction velocity', 'f4', ('lat', 'lon',))
    fricV_nc.units = 'm/s'
    fricT_nc = ds.createVariable('Friction velocity threhold', 'f4', ('lat', 'lon',))
    fricT_nc.units = 'm/s'
    press_nc = ds.createVariable('Surface pressure', 'f4', ('lat', 'lon',))
    press_nc.units = 'Hectopascals'
    temp_nc = ds.createVariable('2-m Temperature', 'f4', ('lat', 'lon',))
    temp_nc.units = 'K'
    SEP_nc = ds.createVariable('SEP', 'f4', ('lat', 'lon',))
    SEP_nc.units = '[unitless]'
    K_nc = ds.createVariable('K', 'f4', ('lat', 'lon',))
    K_nc.units = 'm-1'
    erode_nc = ds.createVariable('Erodibility fraction', 'f4', ('lat', 'lon',))
    erode_nc.units = 'fraction'
    soil_nc = ds.createVariable('Soil type', 'i4', ('lat', 'lon',))
    soil_nc.units = '[unitless]'
    land_nc = ds.createVariable('Landuse type', 'i4', ('lat', 'lon',))
    land_nc.units = '[unitless]'
    emitt_nc = ds.createVariable('FricV-T difference', 'f4', ('lat', 'lon',))
    emitt_nc.units = '[m/s]'

    #Store our saved values into the netcdf file
    lats[:] = np.unique(yi)
    lons[:] = np.unique(xi)
    pm25_nc[:, :] = pm25_flux
    pm10_nc[:, :] = pm10_flux
    fricV_nc[:, :] = fricV
    fricT_nc[:, :] = fricT
    press_nc[:, :] = press
    temp_nc[:, :] = temp
    SEP_nc[:, :] = SEP
    K_nc[:, :] = K
    erode_nc[:,:] = erode
    soil_nc[:,:] = soil_cat
    land_nc[:,:] = land_cat
    emitt_nc[:,:] = dust_emitter

    #Close the netcdf file
    ds.close()



#End of script



