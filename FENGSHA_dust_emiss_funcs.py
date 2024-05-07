import numpy as np
import xarray as xr
import pandas as pd
import os.path as osp
import re, glob, sys, os
import re
from datetime import datetime, timedelta
import netCDF4 as nc
from scipy.interpolate import griddata

#Individual python functions used process different parts of the FENGSHA dust emission
#model. Mostly deals with stuff such as regridding and so on. These routines also include 
#the following functions:
#   (1) regrid_soilT       - Regrid the soil type from a WPS geo_em* file
#   (2) regrid_landT       - Regrid the landuse type from a WPS geo_em* file
#   (3) regrid_soilF       - Regrid soil fraction data to a common grid
#   (4) fricT_table        - Creates a friction velocity threshold table dependent on soil type
#   (5) compute_erode      - Compute the erodibility fraction for each grid cell
#   (6) assign_fricT       - Assigns friction velocity threshold based on soil and landuse types 
#   (7) find_grib_file     - Checks to see if grib file exits
#   (8) met_regrid         - Regrids meteorological data from hrrr to a common grid
#   (9) adjust_gsl         - Adjusts land/soil info around GSL based on current lake level

def regrid_soilT(ncdf_file,xi,yi):

    #  Inputs for this function:
    #  netcdf_file         - is the WPS netcdf file name for soil top information
    #  xi & yi             - meshgrid object with lat lons for the grid we want to 
    #                        interpolate to

    print('Processing static data: Soil type')

    #Lets open up our WRF geogrid file
    ncdf_data = nc.Dataset(ncdf_file)
    lat2D = np.squeeze(np.array(ncdf_data['XLAT_M']))
    lon2D = np.squeeze(np.array(ncdf_data['XLONG_M']))
    soil_type3D = np.squeeze(np.array(ncdf_data['SOILCTOP']))

    #Set missing values to a negative -1. Easier to work with.
    soil_type3D[soil_type3D > 1] = -1

    #Soil data from WPS geogrid file is in a clunky format, where each slice
    #in the third dimension corresponds to a 0 or 1 mask which tells us
    #whether a grid cell is part of classification assigned to the 3D index.

    #Get dims of 3D soil type array and use this to initialize a blank 2D array
    dims = np.shape(soil_type3D)
    soil_type2D = np.zeros((dims[1], dims[2]))

    #Loop through each each in 3rd dimension and assign '1' masks to the correct
    #soil category.
    for s in range(0, dims[0]):
        hit_indices = np.where(soil_type3D[s,:,:] > 0)
        soil_type2D[hit_indices] = s+1

    #Grid data likes the data as a 2D array with lat lon pairs, so merge our 1-D lat/lons
    lon1D = lon2D.flatten()
    lat1D = lat2D.flatten()
    soil1D = soil_type2D.flatten()
    points = np.vstack((lon1D,lat1D)).transpose()

    #Regrid soil data. Assign negative values to 14, which is water. Water can't emit dust.
    soil_regrid = griddata(points, soil1D, (xi, yi), method='nearest')
    soil_regrid[soil_regrid < 1] = 14
    soil_regrid = soil_regrid.astype(int)

    #Returns regridded soil type data as a 2D array
    return soil_regrid
    

def regrid_landT(ncdf_file,xi,yi):

    #  Inputs for this function:
    #  netcdf_file         - is the WPS netcdf file name for soil top information
    #  xi & yi             - meshgrid object with lat lons for the grid we want to 
    #                        interpolate to

    print('Processing static data: Landuse type')

    #Lets open up our WRF geogrid file
    ncdf_data = nc.Dataset(ncdf_file)
    lat2D = np.squeeze(np.array(ncdf_data['XLAT_M']))
    lon2D = np.squeeze(np.array(ncdf_data['XLONG_M']))
    land_use2D = np.squeeze(np.array(ncdf_data['LU_INDEX']))

    #Grid data likes the data as a 2D array with lat lon pairs, so merge our 1-D lat/lons
    lon1D = lon2D.flatten()
    lat1D = lat2D.flatten()
    soil1D = land_use2D.flatten()
    points = np.vstack((lon1D,lat1D)).transpose()

    #Regrid soil data. 
    landuse_regrid = griddata(points, soil1D, (xi, yi), method='nearest')
    landuse_regrid = landuse_regrid.astype(int)

    #Returns regridded landuse data as a 2D array
    return landuse_regrid


def regrid_soilF(netcdf_file,xi,yi):

    #  Inputs for this function:
    #  netcdf_file         - This is the global GLDA soil fraction netcdf file path
    #  xi & yi             - meshgrid object with lat lons for the grid we want to 
    #                        interpolate to

    print('Processing static data: Soil fraction of silt, clay, and sand')

    #Read in the soil fraction data from our prescribed file
    ncdf_data = nc.Dataset(netcdf_file)
    silt = np.squeeze(np.array(ncdf_data['GLDAS_soilfraction_silt']))
    clay = np.squeeze(np.array(ncdf_data['GLDAS_soilfraction_clay']))
    sand = np.squeeze(np.array(ncdf_data['GLDAS_soilfraction_sand']))

    #Also read in our lat lon coordinates, which are provided as lat lons,
    #so lets convert this to a mesh grid afterwards.
    lon1D = np.array(ncdf_data['lon'])
    lat1D = np.array(ncdf_data['lat'])
    lon2D,lat2D = np.meshgrid(lon1D,lat1D)

    #Set non-soil data from -9999 to zero
    silt[silt<0] = 0
    clay[clay<0] = 0
    sand[sand<0] = 0

    #Flatten our soil data to 1-D array so it can be provided as points
    lon1D = lon2D.flatten()
    lat1D = lat2D.flatten()
    silt1D = silt.flatten()
    clay1D = clay.flatten()
    sand1D = sand.flatten()

    #Pair up points to create a single array
    points = np.vstack((lon1D,lat1D)).transpose()

    #Lets regrid that silt, clay, and sand fractions to our common mesh grid 
    silt_regrid = griddata(points, silt1D, (xi, yi), method='linear')
    clay_regrid = griddata(points, clay1D, (xi, yi), method='linear')
    sand_regrid = griddata(points, sand1D, (xi, yi), method='linear')

    #Return silt, clay, and sand percentages
    return silt_regrid, clay_regrid, sand_regrid


def fricT_table():

    #The columns here correspond to [1] Shrubland, [2] Shrubgrass, and [3] barren.
    #Based on FEGNSHA code in CMAQ. Friction velocities based on field experiement 
    #data from: 
    #              Gillette et al., JGR, 1980 for desert soils
    #              Gillette, JGR, 1988 for Loose Agr. Soils 
    #The following soil types were not measured for desert land (we chose to use
    #agr. data): Sandy Clay Loam, Clay Loam, Sandy Clay, and Silty Clay. Modified values
    #compatiable with both MM5 & NAM. There is no measurement of this value for Silt. 
    #The values for Silt are chosen from Silty Loam since the soil composition is close.
	#Other includes all types higher than 12. The values of Other are too high to allow
	#any dust emission. Playa added based on numbers suggested by Mallia et al. 2017.
    #Non-erodible surfaces set to 9999, which this will never generate dust. 

    fricT = [[0.80, 0.42, 0.28],          #Sand [1]
             [1.00, 0.51, 0.34],          #Loamy sand [2]
             [1.40, 0.66, 0.29],          #Sandy loam [3]
             [1.70, 0.34, 1.08],          #Silt Loam [3]
             [1.70, 0.34, 1.08],          #Silt [5]
             [1.70, 0.49, 0.78],          #Loam [6]
             [1.70, 0.78, 0.78],          #Sandy clay loam [7]
             [1.70, 0.33, 0.64],          #Silty clay loam [8]
             [1.70, 0.71, 0.71],          #Clay loam [9]
             [1.70, 0.71, 0.71],          #Sandy clay [10]
             [1.70, 0.56, 0.56],          #Silty clay [11]
             [1.70, 0.78, 0.54],          #Clay [12]
             [9999, 9999, 9999],          #Organic material [13]
             [9999, 9999, 9999],          #Water [14]
             [9999, 9999, 9999],          #Bedrock [15]
             [9999, 9999, 9999],          #Other (ice) [16]
             [0.34, 0.34, 0.34],          #Playa [17]
             [9999, 9999, 9999],          #Lava [18]
             [0.80, 0.42, 0.28]]          #White sand [19]

    fricT = np.array(fricT)
    return fricT


def landU_table():

    #Defines the landuse as either Shrubland [1], Shrubgrass [2], barren [3], or
    #none of these, which is assumed to be not erodible [-1].
    landuse_table = [-1,    #[1] Urban and Built-Up Land 
                     -1,    #[2] Dryland Cropland and Pasture
                     -1,    #[3] Irrigated Cropland and Pasture
                     -1,    #[4] Mixed Dryland/Irrigated Cropland and Pasture
                     -1,    #[5] Cropland/Grassland Mosaic
                     -1,    #[6] Cropland/Woodland Mosaic
                     -1,    #[7] Grassland
                      1,    #[8] Shrubland
                      1,    #[9] Mixed Shrubland/Grassland
                     -1,    #[10] Savanna
                     -1,    #[11] Deciduous Broadleaf Forest
                     -1,    #[12] Deciduous Needleleaf Forest
                     -1,    #[13] Evergreen Broadleaf Forest
                     -1,    #[14] Evergreen Needleleaf Forest
                     -1,    #[15] Mixed Forest
                     -1,    #[16] Water Bodies
                     -1,    #[17] Herbaceous Wetland
                     -1,    #[18] Wooded Wetland
                      3,    #[19] Barren or Sparsely Vegetated
                     -1,    #[20] Herbaceous Tundra
                     -1,    #[21] Wooded Tundra
                     -1,    #[22] Mixed Tundra
                     -1,    #[23] Bare Ground Tundra
                     -1,    #[24] Snow or Ice
                      3,    #[25] Playa
                     -1,    #[26] Lava
                      3,    #[27] White Sand
                     -1,    #[28] Unassigned
                     -1,    #[29] Unassigned
                     -1,    #[30] Unassigned
                     -1,    #[31] Low Intensity Residential
                     -1,    #[32] High Intensity Residential
                     -1]    #[33] Industrial or Commercial

    landuse_table = np.array(landuse_table)
    landuse_table = landuse_table.astype(int)

    #Return our landuse table that defines landuse types as shrubland, shrubgrass, 
    #barren, or none.
    return landuse_table


def assign_fricT(frict_table,landuse_table,land_cat,soil_cat):

    #  Inputs for this function:
    #  netcdf_file         - A table with fiction velocity thresholds for each soil type
    #  landuse_table       - A table that specifies whether a land type is Shrubland [1], 
    #                        Shrubgrass [2], barren [3], or none of these.
    #  land_cat            - 2D grid of landuse information
    #  soil_cat            - 2D grid of soil type information

    print('Processing static data: Computing friction velocity threshold') 

    #Define dimensions and create an empty array to fill in
    dims = np.shape(soil_cat)
    fricT2D = np.zeros(dims)

    #Loop through each possible soil type in the USGS database (1-20) and landuse type so we can
    #assign a friction velocity based on the lookup table defined via frict_table.
    for s in range(0,np.shape(frict_table)[0]):
        for l in range(0,len(landuse_table)):

            #What gridcells equal the soil and landuse type we are working on?
            my_indices = np.where((soil_cat==s+1)&(land_cat==l+1))

            #If we have a non-erodible surface defined as -1, set the friction velocity
            #threshold to an impossibly high number otherwise assign this value with our
            #friction velocity threshold lookup table.
            isit_barren = landuse_table[l]

            if(isit_barren==-1):
                fricT2D[my_indices] = 9999
            else:
                fricT2D[my_indices] = frict_table[s,isit_barren-1]
        
    #Return the 2D array of friction velocity thresholds
    return fricT2D


def compute_erode(land_cat,landuse_table):

    #  Inputs for this function:
    #  land_cat         - Regridded landuse array
    #  landuse_table    - landuse table that will convert general landuse categories into 
    #                   - FENGSHA recognized categories

    print('Processing static data: Computing erodibility')

    #Assign landuse category as new erodibility array
    erode = land_cat.astype(np.float64)

    #Loop through each landuse category and determine where its defined as Shrubland [1], Shrubgrass [2], 
    #barren [3], and none of the above [-1] by FENGSHA
    for l in range(0,len(landuse_table)):
        erode[land_cat==l+1] = landuse_table[l]

    #Determine the erodibility factor based on the landuse type
    erode[erode==-1] = 0
    erode[erode==1] = .5
    erode[erode==2] = .25
    erode[erode==3] = .75

    #Return the 2D erodibility matrix
    return erode


def find_grib(hrrr_path, emiss_time):

    #  Inputs for this function:
    #  hrrr_path         - The path where our hrrr files reside
    #  emiss_time        - The time that we are processing for our dust emissions

    #Convert date object into string for YYYYMMDDHH
    etime = str(emiss_time)
    YYYY = etime[0:4]
    MM = etime[5:7]
    DD = etime[8:10]
    HH = etime[11:13]
    YYYYMMDD = str(YYYY+MM+DD)

    #Form path for grib file
    grib_file = hrrr_path+YYYYMMDD+'/fvel_hrrr.t'+HH+'z.wrfsfcf00.grib2'

    #Does the grib file exit?
    file_check = os.path.exists(grib_file)

    #If not, go back one hour and use the 1 hour forecast (f01) if it exists. Most HRRR 
    #data outages were for single runs. 
    if file_check == False:

        print('No f00 grib file found, try f01 grib file')
        emiss_time = emiss_time - timedelta(hours=1)
        etime = str(emiss_time)

        #Convert date object into string for YYYYMMDDHH
        YYYY = etime[0:4]
        MM = etime[5:7]
        DD = etime[8:10]
        HH = etime[11:13]
        YYYYMMDD = str(YYYY+MM+DD)

        #Form path for grib file
        grib_file = hrrr_path+YYYYMMDD+'/fvel_hrrr.t'+HH+'z.wrfsfcf01.grib2'

        #Does the grib file exit?
        file_check = os.path.exists(grib_file)

    #Return path of file and whether we found either a f00 or f01 file.
    return file_check, grib_file


def met_regrid(var, lon2D, lat2D, xi, yi):

    #  Inputs for this function:
    #  var              - The meteorological variable that we want to regrid
    #  lon2D & lat2D    - meshgrid object of the original lat and lon of met variable
    #  xi & yi          - meshgrid object of lat lon we want to interpolate meteorology to

    #Subset the HRRR data based on the domain we are interested in as defined by the
    #domain extent input variable. First, determine indices to subset
    lon_min = np.round(np.min(xi),5)
    lon_max = np.round(np.max(xi),5)
    lat_min = np.round(np.min(yi),5)
    lat_max = np.round(np.max(yi),5)

    index_check = np.array(np.where((lon2D>lon_min)&(lat2D>lat_min)&(lon2D<lon_max)&(lat2D<lat_max)))
    i_min = np.min(index_check[1,:])
    i_max = np.max(index_check[1,:])
    j_min = np.min(index_check[0,:])
    j_max = np.max(index_check[0,:])

    #Now subset the array now that we know the indices we want to subset. Do this for the
    #meteorological variable and its lat/lon coordinate array
    var2D = var[j_min:j_max,i_min:i_max]
    lon2D = lon2D[j_min:j_max,i_min:i_max]
    lat2D = lat2D[j_min:j_max,i_min:i_max]

    #Ok need to flatten our array since griddata is going to treat on lambert conformal
    #grid as unstructured data. So we will provided the data as points with a lat lon.
    var1D = var2D.flatten()
    lon1D = lon2D.flatten()
    lat1D = lat2D.flatten()

    #Grid data likes the data as a 2D array with lat lon pairs, so merge our 1-D lat/lons
    points = np.vstack((lon1D,lat1D)).transpose()

    #Regrid meteorological variable to the mesh grid that we defined
    var_regrid = griddata(points, var1D, (xi, yi), method='linear')

    #Returns regridded meteorological data as a 2D array
    return var_regrid


def adjust_gsl(gsl_file, lake_level, land_cat, soil_cat, xi, yi, grid_resolution):

    #  Inputs for this function:
    #  gsl_file         - netcdf file with GSL bathymetry data
    #  lake_level       - lake level information
    #  land_cat         - landuse categories that will be adjust based on lake level
    #  soil_cat         - soil categories that will be adjust based on lake level
    #  xi & yi          - lake level we want to prescribe
    #                     interpolate to
    #  grid_resolution  - resolution of our emission grid

    print('Updating landuse data around the GSL based on prescribed lake level')
    print('Lake level set to = '+str(lake_level)+' mASL')

    #Read in the netcdf file 
    ncdf_data = nc.Dataset(gsl_file)
    lat = np.squeeze(np.array(ncdf_data['latitude']))
    lon = np.squeeze(np.array(ncdf_data['longitude']))
    bath = np.squeeze(np.array(ncdf_data['bathym']))
    gsl_domain = [-113.1,-111.9,40.6,41.75]
    gsl_index = np.array(np.where((xi>gsl_domain[0])&(xi<gsl_domain[1])&(yi>gsl_domain[2])&(yi<gsl_domain[3])))

    #Loop through each index and see if this gridcell has an elevation above or below the set lake 
    #level
    for l in range(0, np.shape(gsl_index)[1]):

        #Determine lat lons we are working within our emissions grid
        lon_check = np.round(xi[gsl_index[0,l],gsl_index[1,l]],3)
        lat_check = np.round(yi[gsl_index[0,l],gsl_index[1,l]],3)
        landuse_check = land_cat[gsl_index[0,l],gsl_index[1,l]]
        
        #Ok, lets find which bathymetry grid cells fall within our emission gridcell, since our emission
        #grid spacing is probably much larger than the bathymetry data set and encompasses many grid cells.
        #We can then average all of these gridcells and use that final number to determine if the gridcell
        #fall above or below the prescribed lake level.
        lon_min = lon_check - (grid_resolution/2)
        lon_max = lon_check + (grid_resolution/2)
        lat_min = lat_check - (grid_resolution/2)
        lat_max = lat_check + (grid_resolution/2)
        x_index = np.where((lon>lon_min)&(lon<lon_max))
        y_index = np.where((lat>lat_min)&(lat<lat_max))
        avg_bath = np.mean(bath[y_index,x_index])
    
        #If the average bathymetry height is above the lake level, and the gridell is assigned as water, 
        #switch our landuse cat to playa. Have the water check in there since we don't want to re-assign
        #something that is assigned to something other than water (unlikely but you never know).
        if((avg_bath>lake_level)&(landuse_check==16)):
            land_cat[gsl_index[0,l],gsl_index[1,l]] = 25
            soil_cat[gsl_index[0,l],gsl_index[1,l]] = 17

    return land_cat,soil_cat


