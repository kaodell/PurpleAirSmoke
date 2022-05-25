#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_monitor_wSVI.py
    python script to identify SVI for monitor pair location, and plot the distribution
    of population and co-located monitor count by SVI
Created on Sun Apr 25 19:35:46 2021
@author: kodell
"""
#%% user inputs
sensor_list_fn = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/PA_data/co_loc_sensor_list_1000m_Bonne_global4_fv.csv'
svi_shape_fn = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/SESdata/SVI from the CDC/SVI2018_US/SVI2018_US_tract'
svi_data_fn = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/SESdata/SVI from the CDC/SVI2018_US.csv'

out_fig_fp = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/PA_figures/final/'
out_fig_desc = 'all_1000_fv'

#%% import modules
import pandas as pd
import numpy as np
import shapefile
import matplotlib as mplt
import matplotlib.pyplot as plt

# set figure fonts
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('sansserif')
font.set_name('Tahoma')

#%% load data
sensor_list = pd.read_csv(sensor_list_fn)
svi_data = pd.read_csv(svi_data_fn)

#%% assign monitors SVI
svi_file = shapefile.Reader(svi_shape_fn)
svi_records = svi_file.records()
svi_shapes = svi_file.shapes()
si = 0
area_fips = [['06075','06081','06097','06055','06095','06013','06001','06085','06041'],
             ['49035',],
             ['06037','06111','06071','06059','06065'],
             ['53033','41051','41005'],
             ['08069','08013','08031','08123','08059','08001','08005','08014','08035','08041',
              #addition CFR counties with no monitors but to be added to SVI check
              '08039','08119','08101']]
areas = ['SF','SLC','LA','PNW','CFR']
FIPS_flag = np.array([0]*sensor_list.shape[0])
n_monitors = np.array([0]*svi_data.shape[0])
area_abbr = np.array(['0000']*svi_data.shape[0])

# loop through SVI shapefile to assign monitor totals and area abbriviations to census tracts
fips_check = []
# create a copy of the list to use in the loop - we are going to delete sites as we assign them to
# census tracts - there were 4 cases of double counting monitors in the SF region
# since the SF region has so many monitors, I don't think this is an issue and changing this
# double counting did not change the results
sensor_list_use = sensor_list.copy()
sensor_list_use['og_inds'] = sensor_list['Unnamed: 0']
for j in range(len(svi_records)):
        FIPs = svi_records[j][5]
        # below only want :5 because these are codes for census tracts,
        # we are matching to counties
        if FIPs[:5] in area_fips[0]:
            area_abbr[j] = areas[0]
        elif FIPs[:5] in area_fips[1]:
            area_abbr[j] = areas[1]
        elif FIPs[:5] in area_fips[2]:
            area_abbr[j] = areas[2]
        elif FIPs[:5] in area_fips[3]:
            area_abbr[j] = areas[3]
        elif FIPs[:5] in area_fips[4]:
            area_abbr[j] = areas[4]
        else:
            area_abbr[j] = 'NAN'

        svi_shp = svi_shapes[j]
        for i in range(len(svi_shp.parts)):
            i0 = svi_shp.parts[i]
            if i < len(svi_shp.parts)-1:
            		i1 = svi_shp.parts[i+1] - 1
            else:
            		i1 = len(svi_shp.points)
            seg = svi_shp.points[i0:i1+1]
            mpath = mplt.path.Path(seg)
            # flag for list
            points = np.array((sensor_list_use['in_lon'].values,
                               sensor_list_use['in_lat'].values)).T
            mask = mpath.contains_points(points)#.reshape(glon.shape)
            county_indsA = np.where(mask==True)[0]
            county_inds = sensor_list_use['og_inds'].iloc[county_indsA].values
            FIPS_flag[county_inds] = int(FIPs)
            n_monitors[j]+= len(county_inds)
            
            # to fix double-counting, delete monitors that have already been used
            sensor_list_use.drop(county_indsA,inplace=True)
            sensor_list_use.reset_index(inplace=True,drop=True)
            
        fips_check.append(int(FIPs))

svi_data['n_mons'] = n_monitors
svi_data['FIPS_check'] = fips_check
svi_data['area_abbr'] = area_abbr

#%% for each area, calc cumulative distribution of each of these as a function of ses and plot
# sort sensor_list by ses
svi_sort = svi_data.sort_values(by='RPL_THEMES')
svi_sort.reset_index(inplace=True,drop=True)

# get total US values
cum_sum_pop = np.cumsum(svi_sort['E_TOTPOP'].values)/np.sum(svi_sort['E_TOTPOP'].values)
cum_sum_monitors = np.cumsum(svi_sort['n_mons'])/np.sum(svi_sort['n_mons'].values)

# plot total US values
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.plot(svi_sort['RPL_THEMES'],100.0*cum_sum_pop,'--',label='Population',color='dimgrey')
ax.plot(svi_sort['RPL_THEMES'],100.0*cum_sum_monitors,label='Monitors',color='dimgrey')

# get area-specific values and plot
county_colors = ['#984ea3','#e41a1c','#ff7f00','#377eb8','#4daf4a']
ai = 0
for area in ['SF','SLC','LA','PNW','CFR']:
    inds = np.where(svi_sort['area_abbr']==area)[0]
    cum_sum_pop = np.cumsum(svi_sort['E_TOTPOP'].iloc[inds].values)/np.sum(svi_sort['E_TOTPOP'].iloc[inds].values)
    cum_sum_monitors = np.cumsum(svi_sort['n_mons'].iloc[inds].values)/np.sum(svi_sort['n_mons'].iloc[inds].values)

    ax.plot(svi_sort['RPL_THEMES'].iloc[inds],100.0*cum_sum_pop,'--',color = county_colors[ai])
    ax.plot(svi_sort['RPL_THEMES'].iloc[inds],100.0*cum_sum_monitors, color=county_colors[ai])
    
    # print values for "high vulneability" group in each area
    ind = np.where(svi_sort['RPL_THEMES'].iloc[inds]>0.5)[0][0]
    cum_sum_monitors = np.array(cum_sum_monitors)
    cum_sum_pop = np.array(cum_sum_pop)
    print(area,'mon:',100.0*(1.0-cum_sum_monitors[ind]),'pop:',100.0*(1.0-cum_sum_pop[ind]))
    ai += 1
# add lables to plot    
ax.text(97,30,'Full US',color='dimgrey',fontsize=16,
        fontweight='bold',ha='right')
ax.text(97,20,'San Francisco',color=county_colors[0],fontsize=16,
        fontweight='bold',ha='right')
ax.text(97,15,'Salt Lake City',color=county_colors[1],fontsize=16,
        fontweight='bold',ha='right')
ax.text(97,10,'Los Angeles',color=county_colors[2],fontsize=16,
        fontweight='bold',ha='right')
ax.text(97,25,'Seattle & Portland',color=county_colors[3],fontsize=16,
        fontweight='bold',ha='right')
ax.text(97,5,'Denver',color=county_colors[4],fontsize=16,
        fontweight='bold',ha='right')
# adjust plot visuals and save
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel('Social Vulnerability Index',fontsize=16,)
ax.set_ylabel('Cumulative Percent',fontsize=16,)
ax.set_xlim(0,1)
ax.set_ylim(0,100)
ax.legend(frameon=False,fontsize=16,)
plt.savefig(out_fig_fp+'SVI_mons_pop'+out_fig_desc+'.png',dpi=300)
fig.show()

# have plotted monitors with SVI shapefile to make sure this works, but removed from this version

#%% check monitor number totals for areas to make sure we got all of them
print('the following pairs of numbers should match')
for area in ['SF','SLC','LA','PNW','CFR']:
    indsA = np.where(svi_sort['area_abbr']==area)[0]
    print(np.sum(svi_sort['n_mons'].iloc[indsA].values))
    indsB = np.where(sensor_list['area_abbr']==area)
    print(len(indsB[0]))
    
          
           
           
           
           
           
           
           