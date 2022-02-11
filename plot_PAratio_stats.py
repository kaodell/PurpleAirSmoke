#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_PAratio_stats.py
    python script to plot purple air ratio statistics from analysis run on remote server
Created on Fri Apr 16 22:04:03 2021
@author: kodell
"""
#%% user inputs
# designate files to load
file_desc = 'fv' # fv = final_version
version = 'main' # do calculation for main or SI version of figures, 
                    #only changes ratio and boxplot stats file needed

# project folder
prj_folder = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/PA_data/'
# files to load
ratio_file_load = 'overall_mon_stats_'+file_desc+'_'+version+'.csv'
sensor_list_fp = 'co_loc_sensor_list_1000m_Bonne_global4_'+file_desc+'_wSES.csv'
cleaning_numbers_fp = 'clean_stats_test_2020_nTRH_wUS_'+file_desc+'.csv'
boxplot_nums_fp = 'daily_area_stats_sflag_update_wUS_'+file_desc+'_'+version+'.csv'
aqi_nums_fp = 'aqi_stats_daily_' # just the first part, have to load by each area below
# shape file for plotting counties in Figure 2
shp_fn = prj_folder + 'cb_2018_us_county_500k/cb_2018_us_county_500k.shp'

# where to put output figures
out_fig_fp = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/PA_figures/final/'
out_fig_desc = 'daily_'+file_desc+'_'+version

#%% load modules
import pandas as pd
import numpy as np

# plotting with plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "chrome"
pio.templates.default = "seaborn"
mapbox_access_token='pk.eyJ1Ijoia2FvZGVsbCIsImEiOiJjanZza3k1bGkzMHZoNDhwYjdybTYyOTliIn0.KJyzHWVzu2U087Ps215_LA'

# plotting with matplotlib
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib
import shapefile

# set figure fonts
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('sansserif')
font.set_name('Tahoma')
COLOR = 'black'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
#%% user-defined functions
def mk_map(ax):
    ax.outline_patch.set_edgecolor('white')
    reader = shpreader.Reader('cb_2018_us_state_20m/cb_2018_us_state_20m.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray',alpha=0.5)
    ax.set_extent([-103,-125,31,49])

#%% load data
ratio_df_all = pd.read_csv(prj_folder + ratio_file_load)
# sensor list
sensor_list = pd.read_csv(prj_folder + sensor_list_fp)
# cleaning stats
clean_nums = pd.read_csv(prj_folder + cleaning_numbers_fp)
# boxplot numbers
bplot_nums = pd.read_csv(prj_folder+boxplot_nums_fp)
#%% clean ratio file
# drop locations with < less than 10 smoke free or smoke-impacted days
drop_inds1 = np.where(ratio_df_all['num_s']<(10))[0]
drop_inds2 = np.where(ratio_df_all['num_ns']<(10))[0]
drop_inds = np.unique(np.hstack([drop_inds1,drop_inds2]))
ratio_df = ratio_df_all.drop(drop_inds)
ratio_df.reset_index(inplace=True,drop=True)

# remove invalid ratio values (denominator, ie outdoor PM, was 0)
keep_inds = np.where(np.isfinite(ratio_df['ratio_median_ns']))[0]
ratio_df = ratio_df.iloc[keep_inds]
ratio_df.reset_index(inplace=True,drop=True)

#%% add SES, county, and area flag to ratio_df all and ratio df 
for ratios in [ratio_df_all,ratio_df]:
    ses, county, area = [], [], []
    for ID in ratios['in_AID_tp']:
        ind = np.where(sensor_list['in_AID_tp']==ID)[0]
        ses.append(sensor_list['RPL_THEMES'].iloc[ind].values[0])
        county.append(sensor_list['county_name'].iloc[ind].values[0])
        area.append(sensor_list['area_abbr'].iloc[ind].values[0])
    ratios['RPL_THEMES'] = ses
    ratios['counties'] = county
    ratios['area_abbr'] = area

#%% panel map of median PM - Figure 1
fig,axarr = plt.subplots(ncols=2,nrows=2,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
titles = np.array([['(a) Outdoor PM$_{2.5}$ smoke-free days','(b) Outdoor PM$_{2.5}$ smoke-impacted days'],
                   ['(c) Indoor PM$_{2.5}$ smoke-free days','(d) Indoor PM$_{2.5}$ smoke-impacted days']])

data = np.array([[ratio_df['oPM_median_ns'].values, ratio_df['oPM_median_si'].values],
                   [ratio_df['inPM_median_ns'].values, ratio_df['inPM_median_si'].values]])
for i in range(2):
    for j in range(2):
        ax = axarr[i,j]
        mk_map(ax)
        cs = ax.scatter(ratio_df['lon'].values,ratio_df['lat'].values, c=data[i,j],
                        transform=ccrs.PlateCarree(),cmap='plasma',vmin=1.0,vmax=50.0,
                        norm=matplotlib.colors.LogNorm())
        cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom',pad=0.0,shrink=0.9)
        cbar=fig.colorbar(cs,cax=cax,extend='max',**kw)
        cbar.set_label('PM$_{2.5}$ [$\mu$g m$^{-3}$]',fontsize=10,fontproperties = font)
        ax.set_title(titles[i,j],fontsize=10,fontproperties = font)
plt.savefig(out_fig_fp+'inout_PM_map_median'+out_fig_desc+'.png',dpi=300)
plt.show()

#%% Figure S4
fig,axarr = plt.subplots(ncols=2,nrows=2,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
titles = np.array([['(a) Indoor/Outdoor PM$_{2.5}$ smoke-free days','(b) Indoor/Outdoor PM$_{2.5}$ smoke-impacted days'],
                   ['(a) Spearman R smoke-free days','(b) Spearman R smoke-impacted days']])
data = np.array([[ratio_df['ratio_mean_ns'].values, ratio_df['ratio_mean_s'].values],
                   [ratio_df['spearman_r_ns'].values, ratio_df['spearman_r_s'].values]])
cmaps = np.array([['bwr','bwr'],['viridis','viridis']])
vmax = np.array([[2,2],[1,1]])
cbar_labels = np.array([['Ratio','Ratio'],['Spearman R', 'Spearman R']])
for i in range(2):
    for j in range(2):
        ax = axarr[i,j]
        mk_map(ax)
        cs = ax.scatter(ratio_df['lon'].values,ratio_df['lat'].values, c=data[i,j],
                        transform=ccrs.PlateCarree(),cmap=cmaps[i,j],vmin=0.0,vmax=vmax[i,i])
        cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom',pad=0.0,shrink=0.9)
        cbar=fig.colorbar(cs,cax=cax,extend='max',**kw)
        cbar.set_label(cbar_labels[i,j],fontsize=10,fontproperties = font)
        ax.set_title(titles[i,j],fontsize=10,fontproperties = font)
plt.savefig(out_fig_fp+'smoke_ratio_corr_map'+out_fig_desc+'.png',dpi=300)
plt.show()

#%% Map of monitors in each of the regions - Figure 2 a-e
county_colors = ['#984ea3','#ff7f00','#377eb8','#e41a1c','#4daf4a']
areas = ['SF','LA','PNW','SLC','CFR']
# to color the correct counties, also need the FIPs codes used for each area
area_fips = [['06075','06081','06097','06055','06095','06013','06001','06085','06041'],
             ['06037','06111','06071','06059','06065'],
             ['53033','41051','41005'],
             ['49035',],
             ['08069','08013','08031','08123','08059','08001','08005','08014','08035','08041',
              #addition CFR counties with no monitors but to be added to SVI check
              '08039','08119','08101']]

fig,axarr = plt.subplots(ncols=5,nrows=1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,3))
axarr = axarr.flatten()
ai = 0
reader2 = shapefile.Reader(shp_fn)
records = reader2.records()
shapes = reader2.shapes()
for area in areas:
    inds = np.where(ratio_df_all['area_abbr']==area)[0]
    inds2 = np.where(ratio_df['area_abbr']==area)[0]
    ax = axarr[ai]
    ax.outline_patch.set_edgecolor('grey')
    reader = shpreader.Reader(shp_fn)
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black',alpha=0.5)
    #color counties
    ri = 0
    for record in records:
        if record[4] in np.array(area_fips[ai]):
            area_county = cfeature.ShapelyFeature([counties[ri]],ccrs.PlateCarree())
            ax.add_feature(area_county, facecolor='none', edgecolor=county_colors[ai],alpha=0.8)
        ri += 1
    # plot all monitors with data for analysis in gray, monitors with sufficient data for averages
    cs = ax.scatter(ratio_df_all['lon'],ratio_df_all['lat'],color='grey',transform=ccrs.PlateCarree(),s=1,alpha=1)
    cs = ax.scatter(ratio_df['lon'].iloc[inds2],ratio_df['lat'].iloc[inds2], color=county_colors[ai],transform=ccrs.PlateCarree(),s=2,alpha=1)
    ai += 1
axarr[0].set_extent([-121,-123.5,35.5,40.1])
axarr[1].set_extent([-116,-119.5,32.5,39])
axarr[2].set_extent([-121.5,-123,45.1,47.9])
axarr[3].set_extent([-111.1,-112.8,39.2,42.4])
axarr[4].set_extent([-104.2,-105.7,38.0,40.8])
fig.tight_layout()
plt.savefig(out_fig_fp+'monitor_map'+out_fig_desc+'.png',dpi=500)

#%%  ratio v ratio plot - Figure 4
fig,ax = plt.subplots(1,1)
ai = 0
for area in ['SF','LA','PNW','SLC','CFR']:
    inds = np.where(ratio_df['area_abbr']==area)[0]
    ax.scatter(ratio_df['ratio_median_ns'].iloc[inds],ratio_df['ratio_median_s'].iloc[inds],
       c=county_colors[ai],marker='o',s=12,zorder=ai+1,alpha=0.5)
    # also add point for median ratio
    ax.scatter(ratio_df['ratio_median_ns'].iloc[inds].median(),ratio_df['ratio_median_s'].iloc[inds].median(),
       c=county_colors[ai],marker='*',edgecolor='black',s=200,zorder=6,alpha=1)
    print(area,ratio_df['ratio_median_ns'].iloc[inds].median(),ratio_df['ratio_median_s'].iloc[inds].median(),
          len(inds))
    ai += 1
ax.hlines(1,0,ratio_df['ratio_median_ns'].max()+0.1,colors='black',linestyles='dashed')
ax.vlines(1,0,ratio_df['ratio_median_s'].max()+0.05,colors='black',linestyles='dashed')
ax.set_xlim(0,3.0) # make sure to check what points are excluded with this zoom!
ax.set_ylim(0,1.75)
ax.set_xlabel('Indoor/outdoor PM ratio\nSmoke-free days')
ax.set_ylabel('Indoor/outdoor PM ratio\nSmoke-impacted days')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# add text for each region
ax.text(2.9,1.57,'San Francisco',color=county_colors[0],fontweight='bold',ha='right')
ax.text(2.9,1.49,'Salt Lake City',color=county_colors[3],fontweight='bold',ha='right')
ax.text(2.9,1.41,'Los Angeles',color=county_colors[1],fontweight='bold',ha='right')
ax.text(2.9,1.65,'Seattle & Portland',color=county_colors[2],fontweight='bold',ha='right')
ax.text(2.9,1.33,'Denver',color=county_colors[4],fontweight='bold',ha='right')

plt.subplots_adjust(top=0.96,bottom=0.15,left=0.13,right=0.98,hspace=0.2,wspace=0.2)
plt.savefig(out_fig_fp+'ratio_v_ratio'+out_fig_desc+'.png',dpi=300)

#%% plot PM, ratio by SES - Figures S10-S13
# ai = 0 is SF, ai=1 is LA
for ai in [0,1]:
    inds = np.where(ratio_df['area_abbr']==areas[ai])
    area_ratio_df = ratio_df.iloc[inds]
    area_ratio_df.reset_index(inplace=True,drop=True)
    
    # first indoor and outdoor PM concentrations
    fig,ax = plt.subplots(2,2,figsize=(5,5))
    ax=ax.flatten()
    data_plot = ['oPM_median_ns','oPM_median_si','inPM_median_ns','inPM_median_si']
    for i in range(4):
        ax[i].scatter(area_ratio_df['RPL_THEMES'],area_ratio_df[data_plot[i]],
                   color=county_colors[ai],s=2,alpha=0.7)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    ax[0].set_title('Smoke Free')
    ax[1].set_title('Smoke Impacted')
    ax[2].set_xlabel('SVI')
    ax[3].set_xlabel('SVI')
    ax[0].set_ylabel('Outdoor PM$_{2.5}$ [$\mu$g m$^{-3}$]')
    ax[2].set_ylabel('Indoor PM$_{2.5}$ [$\mu$g m$^{-3}$]')
    plt.tight_layout()
    plt.savefig(out_fig_fp+'PM_by_SVI_'+areas[ai]+'_'+out_fig_desc+'.png',dpi=300)
    
    # smoke-impacted and smoke-free ratios
    fig,ax = plt.subplots(1,2,figsize=(6,4))
    ax[0].scatter(area_ratio_df['RPL_THEMES'],area_ratio_df['ratio_median_ns'],
                  color=county_colors[ai],s=2,alpha=0.7)
    ax[1].scatter(area_ratio_df['RPL_THEMES'],area_ratio_df['ratio_median_s'],
                  color=county_colors[ai],s=2,alpha=0.7)
    for i in range(2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('SVI')
        ax[i].set_ylim([0,2])
    ax[0].set_ylabel('Indoor/Outdoor PM$_{2.5}$')
    ax[0].set_title('Smoke Free')
    ax[1].set_title('Smoke Impacted')
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(out_fig_fp+'ratio_by_SVI_'+areas[ai]+'_'+out_fig_desc+'.png',dpi=300)

#%% print numbers in paper, cleaning stats
# cleaning stats numbers
clean_inds = []
for ci in range(clean_nums.shape[0]):
    sID = str(clean_nums['ID'].iloc[ci])[:4]
    rind = np.where(ratio_df_all['ID']==float(sID))[0]
    clean_inds.append(ci)

# count stats
print('#### CLEANING STATS ####')
totb4 = np.nansum(clean_nums['nobs_b4'].iloc[clean_inds].values)
print('total b4',totb4)
print('total after',np.nansum(clean_nums['nobs_after'].iloc[clean_inds].values))
print('total pct rmv',100.0*(totb4-np.nansum(clean_nums['nobs_after'].iloc[clean_inds].values))/totb4)
print('total paired',np.nansum(clean_nums['npairs'].iloc[clean_inds].values))
print('n > 500 rmv',np.nansum(clean_nums['nPMhigh_rm'].iloc[clean_inds].values),
      '(',100.0*round(np.nansum(clean_nums['nPMhigh_rm'].iloc[clean_inds].values)/totb4,4),'%)')
print('n AB rmv',np.nansum(clean_nums['nPMrm'].iloc[clean_inds].values),
      '(',100.0*round((np.nansum(clean_nums['nPMrm'].iloc[clean_inds].values))/totb4,4),'%)')
print('n T rmv',np.nansum(clean_nums['nTArm'].iloc[clean_inds].values),
      '(',100.0*round((np.nansum(clean_nums['nTArm'].iloc[clean_inds].values))/totb4,4),'%)')
print('n RH rmv',np.nansum(clean_nums['nRHArm'].iloc[clean_inds].values),
      '(',100.0*round((np.nansum(clean_nums['nRHArm'].iloc[clean_inds].values))/totb4,4),'%)')
print('<50% of day rmv',ratio_df_all['n_count_rmv'].sum(),100.0*ratio_df_all['n_count_rmv'].sum()/ratio_df_all['total_obs_b4'].sum())
#%% number of smoke-impacted and smoke-free obs
print('#### Smoke Days and Nonsmoke Day Counts ####')
print('n smoke obs',np.sum(ratio_df_all['num_s']))
print('n nosmoke obs',np.sum(ratio_df_all['num_ns']))

#%% total monitors in each area with 2020 paired data
total_region_mons = 0
for area in areas:
    count =  len(np.where(ratio_df_all['area_abbr']==area)[0])
    total_region_mons += count
    print(area,'n mons 4 ratio:',count)
print('totals','n mons:',total_region_mons)

#%% figure 1 discussion numbers
# difference in indoor PM on smoke-impacted and smoke-free days
print('#### FIGURE 1 DISCUSSION ####')
in_diff_smoke_days = ratio_df['inPM_median_si'].values - ratio_df['inPM_median_ns'].values
finite_inds = np.where(ratio_df['inPM_median_ns']>0) # have to take out 0's for percent difference calc
in_pdiff_smoke_days = (100.0*in_diff_smoke_days[finite_inds])/ratio_df['inPM_median_ns'].iloc[finite_inds].values
print('inPM si - inPM ns')
print('25th, 50th, and 75th percentiles, diff',np.percentile(in_diff_smoke_days,25),
      np.median(in_diff_smoke_days),np.percentile(in_diff_smoke_days,75))
print('25th, 50th, and 75th percentiles, % diff',np.percentile(in_pdiff_smoke_days,25),
      np.median(in_pdiff_smoke_days),np.percentile(in_pdiff_smoke_days,75))

# difference in indoor and outdoor PM on smoke-impacted and smoke-free days
inout_diff_si = ratio_df['inPM_median_si'].values - ratio_df['oPM_median_si'].values
print('N PM out > in, si',len(np.where(inout_diff_si<0)[0]),'of',len(inout_diff_si))
inout_diff_sf = ratio_df['inPM_median_ns'].values - ratio_df['oPM_median_ns'].values
print('N PM out > in, sf',len(np.where(inout_diff_sf<0)[0]))
print('N PM out < in, sf',len(np.where(inout_diff_sf>0)[0]))

# in out % diff on smoke-impacted and smoke-free days
inout_pdiff_si = -100.0*inout_diff_si/ratio_df['oPM_median_si'].values
inout_pdiff_sf = -100.0*inout_diff_sf/ratio_df['oPM_median_ns'].values

print('(out-in)/out smoke impacted 25th, 50th, and 75th percentiles',np.percentile(inout_pdiff_si,25),
      np.median(inout_pdiff_si),np.percentile(inout_pdiff_si,75))
print('(out-in)/out smoke free 25th, 50th, and 75th percentiles',np.percentile(inout_pdiff_sf,25),
      np.median(inout_pdiff_sf),np.percentile(inout_pdiff_sf,75))

# correlation
print('spearman-r si',np.median(ratio_df['spearman_r_s'].values))
print('spearman-r sf',np.median(ratio_df['spearman_r_ns'].values))
diff_r = ratio_df['spearman_r_s'].values - ratio_df['spearman_r_ns'].values
print('25th, 50th, 75th percentile spearmanR diff si-sf',np.percentile(diff_r,25),
      np.median(diff_r),np.percentile(diff_r,75))
print('r sf > r si',len(np.where(diff_r<0)[0]))

#%% figure 3 discussion
print('#### FIGURE 3 DISCUSSION ####')
#bplot_nums.set_index('area',inplace=True)
bplot_nums['PM25_bj_inns med']
bplot_nums['PM25_bj_outns med']
pct_diff_si = 100.0*(bplot_nums['PM25_bj_outs med'].values - bplot_nums['PM25_bj_ins med'].values)/bplot_nums['PM25_bj_outs med'].values 
print('smoke impacted out-in % diff',bplot_nums.index,pct_diff_si)
pct_diff_sf = 100.0*(bplot_nums['PM25_bj_outns med'].values - bplot_nums['PM25_bj_inns med'].values)/bplot_nums['PM25_bj_outns med'].values 
print('smoke free out-in % diff',bplot_nums.index,pct_diff_sf)

bplot_nums['PM25_bj_outs med']
bplot_nums['PM25_bj_ins med']

pct_diff_si = 100.0*(bplot_nums['PM25_bj_ins med'].values - bplot_nums['PM25_bj_inns med'].values)/bplot_nums['PM25_bj_inns med'].values 
print(bplot_nums.index,pct_diff_si)

#%% print numbers in paper, Figure 4 discussion
print('#### FIGURE 4 DISCUSSION ####')
# monitors in the four cities with si ratios < 1
less_1_inds_si = np.where(ratio_df['ratio_median_s']<1)[0]
less_1_inds_sf = np.where(ratio_df['ratio_median_ns']<1)[0]

# monitors with ratios < 1, smoke impacted
total_l1_si = 0
total_l1_sf = 0
total_region_mons = 0
for area in areas:
    count =  len(np.where(ratio_df['area_abbr']==area)[0])
    count_l1_si = len(np.where(ratio_df['area_abbr'].iloc[less_1_inds_si]==area)[0])
    count_l1_sf = len(np.where(ratio_df['area_abbr'].iloc[less_1_inds_sf]==area)[0])
    total_l1_si += count_l1_si
    total_l1_sf += count_l1_sf
    total_region_mons += count
    print(area,'n mon pairs w data:',count,', n ratio <1 sf:',count_l1_sf,'(',round(100.0*count_l1_sf/count,1),'%)'
          ', n ratio <1 si:',count_l1_si,'(',round(100.0*count_l1_si/count,1),'%)')
print('totals','n mons:',total_region_mons,', n ratio <1, sf:',total_l1_sf,', n ratio<1 si:',total_l1_si)

sea_inds = np.where(ratio_df['area_abbr']=='PNW')
print('max si ratio PNW',np.max(ratio_df['ratio_median_s'].iloc[sea_inds]))

#%% Figure 5 discussion
ai = 0
aqi_in_change = []
aqi_ratio_change = []
for area in areas:
    area_aqi_nums = pd.read_csv(prj_folder+aqi_nums_fp+area+'_2020_'+file_desc+'.csv')
    area_aqi_nums['area'] = [area]*area_aqi_nums.shape[0]
    area_aqi_nums.drop('Unnamed: 0',axis=1,inplace=True)
    
    # drop rows with no data
    inds_drop = np.where(area_aqi_nums['n_mon_in']==0)[0]
    area_aqi_nums.drop(inds_drop,axis=0,inplace=True)
    
    # calc change per aqi bin and save
    aqi_in_change.append(np.mean(100.0*np.diff(area_aqi_nums['med_in'].values)/area_aqi_nums['med_in'].iloc[:-1]))
    aqi_ratio_change.append(np.mean(100.0*np.diff(area_aqi_nums['med_ratio'].values)/area_aqi_nums['med_ratio'].iloc[:-1]))

    if ai == 0:
        aqi_nums = area_aqi_nums.copy()
    else:
        aqi_nums = pd.concat([aqi_nums,area_aqi_nums])
    ai += 1
aqi_nums.reset_index(inplace=True,drop=True)

aqi_nums[['stat','n_mon_in','area']]

print('mean inPM change by aqi bin:',np.mean(aqi_in_change).round(2))
print('mean ratio change by aqi bin:',np.mean(aqi_ratio_change).round(2))

#%% plot graveyard
# interactive AGU plots
'''
import chart_studio as cs
import chart_studio.plotly as csp
from plotly.subplots import make_subplots
cs.tools.set_credentials_file(username='kaodell', api_key='PE6xifRj40nEff79lW1C')
fig = make_subplots(rows=1, cols=1,specs=[[{"type": "mapbox"}]])
fig.update_mapboxes(style='open-street-map',center=dict(lat=40.92,lon=-113.07),pitch=0,zoom=4.1)
fig.add_trace(go.Scattermapbox(
    lat=ratio_df['lat'],lon=ratio_df['lon'],mode='markers',
    hovertext = ratio_df['inPM_mean_si'],
    marker=go.scattermapbox.Marker(size=12,color = ratio_df['inPM_mean_si'],
                                   cmin=0,cmax=35,
                                   colorbar=dict(title='mean inPM smoke days'),
                                   colorscale='Magma')),row=1,col=1)
fig.update_layout(title = 'Indoor PM2.5 on smoke-impacted days')
csp.plot(fig)
fig = make_subplots(rows=1, cols=1,specs=[[{"type": "mapbox"}]])
fig.update_mapboxes(style='open-street-map',center=dict(lat=40.92,lon=-113.07),pitch=0,zoom=4.1)
fig.add_trace(go.Scattermapbox(
    lat=ratio_df['lat'],lon=ratio_df['lon'],mode='markers',
    hovertext = ratio_df['ratio_mean_s'],
    marker=go.scattermapbox.Marker(size=12,color = ratio_df['ratio_mean_s'],
                                   cmin=0,cmax=2,
                                   colorbar=dict(title='mean in/out PM smoke days'),
                                   colorscale='rdbu_r')),row=1,col=1)
fig.update_layout(title = 'Indoor/Outdoor PM2.5 on smoke-impacted days')
csp.plot(fig)
'''

