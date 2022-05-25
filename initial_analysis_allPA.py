# initial_analysis_allPA.py
#   python script to make figures of the PA indoor-outdoor comparison by regions
# written by: Katelyn O'Dell
# v0 - 03.24.21 - initial scripting

########################################################################################################
# user inputs
########################################################################################################
file_desc = 'fv' # fv = final version
scratch_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/'
PA_data_fp = scratch_fp + 'ABmrg_clean_wsmoke/'
PAclean_metadata_fn = scratch_fp + 'ABmrg_clean/clean_stats_test_2020_nTRH_wUS_'+file_desc+'.csv'
PAall_metadata_fn = scratch_fp + 'ABmrg/PAmetadata_wprocessed_names_wUScoloc_d1000m_'+file_desc+'.csv'

# metadata with county flags
PAall_metadata_wcounty_flags_fn = '/fischer-scratch/kaodell/purple_air/sensor_lists/wUS_co_loc_sensor_list_1000m_Bonne_global4_'+file_desc+'.csv'

# load daily or hourly?
avg = 'daily'
year = '2020'

# area to load
ai = 0 #corresponds to code and names below
area_abbrs = ['SF','LA','SLC','PNW','CFR']
area_names = ['San Francisco','Los Angeles','Salt Lake City','Seattle & Portland','Denver']

# select pm correction factor to use
ipm_cf = 'PM25_bj'
opm_cf = 'PM25_bj'

# where to put out figures and files
out_fig_fp = '/home/kaodell/NSF/purple_air_smoke/Bonne_data/figures/'
out_file_fp = '/home/kaodell/NSF/purple_air_smoke/'

# labels for output figures, determiend from above inputs
out_desc = avg+'_'+area_abbrs[ai]+'_'+year+'_'+file_desc
area_name = area_names[ai]+', '+year+', '+avg

########################################################################################################
# load modules
########################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy
import os
from scipy.stats import linregress
import datetime as dt
import matplotlib.colors as colors
import scipy.stats as st

########################################################################################################
# load meta data
########################################################################################################
md_df_clean = pd.read_csv(PAclean_metadata_fn)
md_df_all = pd.read_csv(PAall_metadata_fn)
md_df_county_flags = pd.read_csv(PAall_metadata_wcounty_flags_fn)

# add processed names to og metadata
processed_names = []
for ID in md_df_county_flags['in_AID_tp']:
    ind = np.where(md_df_all['in_AID_tp']==ID)[0]
    processed_names.append(md_df_all['processed_names'].iloc[ind].values[0])
md_df_county_flags['processed_names']=processed_names

########################################################################################################
# load in/out monitors, combine data
########################################################################################################
ipm_use = ipm_cf + '_in'
opm_use = opm_cf + '_out'

# identify IDs to load
inds = np.where(md_df_county_flags['area_abbr']==area_abbrs[ai])[0]
IDs_load = md_df_county_flags['processed_names'].iloc[inds].values

# load identified files and stack
count = 0
nodata = 0
nofile = 0
for ID in IDs_load:
    if np.isnan(ID):
        nodata += 1
        continue
    fn_load = 'purpleair_PMAB20_mID'+str(int(ID)).zfill(4)+'_AnBmrg_wUScoloc_d1000m_clean_wsmoke_'+year+'_'+file_desc+'_'+avg+'_.csv'
    if fn_load not in os.listdir(PA_data_fp):
        nofile += 1
        continue
    pa_df = pd.read_csv(PA_data_fp + fn_load)
    pa_df['ID'] = [str(int(ID)).zfill(4)]*pa_df.shape[0]

    # stack monitors
    if count == 0:
        pa_df_all1 = pa_df
    else:
        pa_df_all1 = pd.concat([pa_df_all1,pa_df])
    count += 1
    
print(len(IDs_load)-(nodata+nofile),'monitors available in selected area')
pa_df_all1.reset_index(inplace=True,drop=True)
print('HMS_check',np.unique(pa_df_all1['HMS_flag']))

###########################################################################################
# remove nans and daily averages without enough points
###########################################################################################
naninds_in = np.where(np.isnan(pa_df_all1[ipm_use]))
naninds_out = np.where(np.isnan(pa_df_all1[opm_use]))
naninds = np.unique(np.hstack([naninds_in,naninds_out]))
pa_df_nn1 = pa_df_all1.drop(naninds)
pa_df_nn1.reset_index(inplace=True,drop=True)

# determine number of obs needed based on average selected
if avg == 'daily':
    factor = 24
elif avg == 'hourly':
    factor = 1
else:
    print('averaging method not recognized')
    
# we want at least 50% of the day or hour to have observations, these are 10-minute obs, so should be 6 per hour
daily_inds_rmv1 = np.where(pa_df_nn1['pm_in_count']<0.5*(6*factor))[0]
daily_inds_rmv2 = np.where(pa_df_nn1['pm_out_count']<0.5*(6*factor))[0]
print(100.0*len(daily_inds_rmv1)/pa_df_nn1.shape[0],'percent in obs rmv by 50% criteria')
print(100.0*len(daily_inds_rmv2)/pa_df_nn1.shape[0],'percent out obs rmv by 50% criteria')
daily_inds_rmv = np.unique(np.hstack([daily_inds_rmv1,daily_inds_rmv2]))
print('total rmv by 50% criteria',len(daily_inds_rmv),'total obs (no nans)',pa_df_nn1.shape[0])
pa_df_nn = pa_df_nn1.drop(daily_inds_rmv)
pa_df_nn.reset_index(inplace=True,drop=True)

###########################################################################################
# add flags for seasons and identify smoke inds
###########################################################################################
# add smoke season and general season flag
ss_flag = []
seas_flag = []
for i in range(pa_df_nn.shape[0]):
    month = pa_df_nn['time_loc'].iloc[i].split('-')[1]
    # ultimately didn't use this one
    if month in ['05','06','07','08','09','10','11']:
        ss_flag.append(1)
    else:
        ss_flag.append(0)
    if month in['12','01','02']:
        seas_flag.append(0)
    elif month in ['03','04','05']:
        seas_flag.append(1)
    elif month in ['06','07','08']:
        seas_flag.append(2)
    elif month in ['09','10','11']:
        seas_flag.append(3)
    else:
        seas_flag.append(4) # just to test this is working

pa_df_nn['season_flag']=seas_flag
pa_df_nn['smoke_season_flag']=ss_flag

smoke_inds = np.where(pa_df_nn['smoke_flag']==1)
# test only using HMS for smoke days
#smoke_inds = np.where(pa_df_nn['HMS_flag']==1)
nsmoke_inds = np.where(pa_df_nn['HMS_flag']==0)
nsmoke_inds1 = np.array(nsmoke_inds[0])

# round PM to 2 decimal places, like in the original csvs from PurpleAir and make negatives zero
pa_df_nn[ipm_use] = np.round(pa_df_nn[ipm_use].values,2)
pa_df_nn[opm_use] = np.round(pa_df_nn[opm_use].values,2)
pa_df_nn[ipm_use] = np.where(pa_df_nn[ipm_use]<0,0,pa_df_nn[ipm_use])
pa_df_nn[opm_use] = np.where(pa_df_nn[opm_use]<0,0,pa_df_nn[opm_use])

# finally, add ratios to dataframe
pa_df_nn['ratio'] = pa_df_nn[ipm_use]/pa_df_nn[opm_use]

###########################################################################################
# get summer and fall smoke-impacted inds for plotting AQI figure
###########################################################################################
jja_inds = np.where(pa_df_nn['season_flag']==2)[0]
son_inds = np.where(pa_df_nn['season_flag']==3)[0]

jja_s_inds = jja_inds[np.where(pa_df_nn['smoke_flag'].iloc[jja_inds]==1)[0]]
son_s_inds = son_inds[np.where(pa_df_nn['smoke_flag'].iloc[son_inds]==1)[0]]

###########################################################################################
# Figure 1: check cf1/atm labels are correct
###########################################################################################
pm_atm = np.hstack([pa_df_nn['PMatm_in'].values,pa_df_nn['PMatm_out'].values])
pm_cf1 = np.hstack([pa_df_nn['PMcf_1_avg_in'].values,pa_df_nn['PMcf_1_avg_out'].values])
fix,ax = plt.subplots(ncols=1,nrows=1)
ax.scatter(pm_atm,pm_cf1)
ax.set_xlabel('atm')
ax.set_ylabel('cf1')
ax.plot([0,250],[0,250],'k')
ax.plot([0,166.67],[0,250],'r') # atm should be ~ 2/3 of CF1 above a certain concentration
#plt.show()

###########################################################################################
# Figure 2: plot monitors on a map to ensure we have the right area
###########################################################################################
lons = []
lats = []
for ID in IDs_load:
    ind = np.where(md_df_all['processed_names']==ID)[0]
    if len(ind)==0:
        continue
    lons.append(md_df_all['in_lon'].iloc[ind].values[0])
    lats.append(md_df_all['in_lat'].iloc[ind].values[0])
fig,ax = plt.subplots(ncols=1,nrows=1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
ax.patch.set_visible(False)
ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='gray',alpha=0.5)
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'))
ax.coastlines('50m')
ax.outline_patch.set_edgecolor('white')
cs = ax.scatter(lons,lats,transform=ccrs.PlateCarree(),color='red')
plt.tight_layout()
#plt.show()

########################################################################################################
# Figure 3: time period covered by selected monitors
########################################################################################################
fig,ax = plt.subplots(1,1)
ax.scatter(pd.to_datetime(pa_df_nn['time_loc'].values),pa_df_nn['ID'],c=pa_df_nn['HMS_flag'],vmin=-1,vmax=1,s=1)
#plt.show()

########################################################################################################
# Figure 4: 1-1 in v out, smoke impacted and smoke free histogram (hourly version in supplement)
########################################################################################################
fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
ax = ax.flatten()
bins = [np.logspace(0,2.7,100,base=10),np.logspace(0,2.7,100,base=10)] # 10**2.7 = 500, max PM we should have here
counts, xedges, yedges, im0 = ax[0].hist2d(pa_df_nn[opm_use].iloc[nsmoke_inds].values,pa_df_nn[ipm_use].iloc[nsmoke_inds].values,bins=bins,cmap='BuPu', norm=colors.LogNorm())
counts, xedges, yedges, im1 = ax[1].hist2d(pa_df_nn[opm_use].iloc[smoke_inds].values,pa_df_nn[ipm_use].iloc[smoke_inds].values,bins=bins,cmap='BuPu',norm=colors.LogNorm())
ax[0].set_xlabel('Outside PM$_{2.5}$ [$\mu$g m$^{-3}$]')
ax[1].set_xlabel('Outside PM$_{2.5}$ [$\mu$g m$^{-3}$]')
ax[0].set_ylabel('Inside PM$_{2.5}$ [$\mu$g m$^{-3}$]')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].plot([10**0,10**2.7],[10**0,10**2.7],'--',c='black')
ax[1].plot([10**0,10**2.7],[10**0,10**2.7],'--',c='black')
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[0].title.set_text('Smoke-Free Days')
ax[1].title.set_text('Smoke-Impacted Days')
fig.colorbar(im0,ax=ax[0])
fig.colorbar(im1,ax=ax[1])
fig.suptitle(area_name)
plt.savefig(out_fig_fp + 'in-v-out_hist_'+out_desc+'.png')
#fig.show()

######################################################################################
# Figure 5: inside PM and ratio as a function of AQI
######################################################################################

pa_df_ss = pa_df_nn.iloc[np.hstack([jja_s_inds,son_s_inds])]
pa_df_ss.reset_index(inplace=True,drop=True)
pm_use = 'Smoke-Impacted'

for pa_df_use in [pa_df_ss]:
    # get inds for each AQI category
    good_inds = np.array(np.where(pa_df_use[opm_use]<12.05)[0])
    mod_inds1 = np.array(np.where(np.logical_and(pa_df_use[opm_use]>=12.05,pa_df_use[opm_use]<35.45))[0])
    usi_inds = np.array(np.where(np.logical_and(pa_df_use[opm_use]>=35.45,pa_df_use[opm_use]<55.45))[0])
    u_inds = np.array(np.where(np.logical_and(pa_df_use[opm_use]>=55.45,pa_df_use[opm_use]<150.45))[0])
    vh_inds = np.array(np.where(np.logical_and(pa_df_use[opm_use]>=150.45,pa_df_use[opm_use]<250.45))[0])
    hazd_inds = np.where(pa_df_use[opm_use]>=250)[0]
    
    # remove really low points in SF smoke-impacted for figure clarity and print values to add to caption
    mod_rmv_inds = np.where(pa_df_use[ipm_use].iloc[mod_inds1]<1.0)
    mod_inds = np.delete(mod_inds1,mod_rmv_inds)
    print('low inds removing:',len(mod_rmv_inds[0]),pa_df_use[ipm_use].iloc[mod_inds1[mod_rmv_inds]])

    # make arrays of data for plotting and logging data
    # plotting
    bp1_data = []
    bp2_data = []
    # logging data
    med_in = []
    lqt_in = []
    uqt_in = []
    med_ratio = []
    lqt_ratio = []
    uqt_ratio = []
    n_mon_in = []
    for inds in [good_inds,mod_inds,usi_inds,u_inds,vh_inds,hazd_inds]:
        bp1_data.append(pa_df_use[ipm_use].iloc[inds])
        bp2_data.append(pa_df_use['ratio'].iloc[inds])
    # for logging data we want all the moderate inds
    for inds in [good_inds,mod_inds1,usi_inds,u_inds,vh_inds,hazd_inds]:
        n_mon_in.append(len(np.unique(pa_df_use['ID'].iloc[inds])))
        if len(inds) == 0:
            med_in.append(-1)
            lqt_in.append(-1)
            uqt_in.append(-1)
            med_ratio.append(-1)
            lqt_ratio.append(-1)
            uqt_ratio.append(-1)
        else:
            med_in.append(pa_df_use[ipm_use].iloc[inds].median())
            lqt_in.append(np.percentile(pa_df_use[ipm_use].iloc[inds],25))
            uqt_in.append(np.percentile(pa_df_use[ipm_use].iloc[inds],75))
            med_ratio.append(pa_df_use['ratio'].iloc[inds].median())
            lqt_ratio.append(np.percentile(pa_df_use['ratio'].iloc[inds],25))
            uqt_ratio.append(np.percentile(pa_df_use['ratio'].iloc[inds],75))
    
    # make figure
    fig, axs = plt.subplots(2,1,figsize=(10,6))
    axs.flatten()
    box_labels = ['<12','12-35','35-55','55-150','150-250','>250']
    bp1 = axs[0].boxplot(bp1_data,labels=box_labels,whis=(0,100),patch_artist=True,showfliers=True)
    bp2 = axs[1].boxplot(bp2_data,labels=box_labels,whis=(0,100),patch_artist=True,showfliers=True)

    # add 1 line and colors in the back for inside AQI
    colors = ['green','gold','darkorange','red','blueviolet','firebrick']
    bin_edges = [0,12,35,55,150,250]
    for i in range(len(colors)-1):
        axs[0].axhspan(bin_edges[i],bin_edges[i+1],color=colors[i],alpha=0.2)
    axs[1].axhline(1,linestyle='--',color='black')

    # add N inside each box
    inds_list = [good_inds,mod_inds1,usi_inds,u_inds,vh_inds,hazd_inds]
    for ni in range(len(inds_list)):
        axs[0].text(ni+1,250,'N: '+str(len(inds_list[ni])),ha='center')
    
    # color box attributes and set axes formats
    for bplot in (bp1,bp2):
        for i in range(len(bplot['medians'])):
            bplot['medians'][i].set(color='black')
            bplot['boxes'][i].set(color=colors[i],edgecolor='black')
            for var in ['caps','whiskers']:
                bplot[var][2*i].set(color='black')
                bplot[var][2*i+1].set(color='black')
        for j in range(2):
            axs[j].spines["right"].set_visible(False)
            axs[j].spines["top"].set_visible(False)
            axs[j].set_yscale('log')

    # add titles, labels, and save
    axs[0].set_ylabel('Inside PM$_{2.5}$ [$\mu$g m$^{-3}$]')
    axs[1].set_ylabel('Inside PM$_{2.5}$/Outside PM$_{2.5}$')
    axs[1].set_xlabel('AQI bin')
    fig.suptitle(area_name+', '+pm_use+' Days')
    plt.savefig(out_fig_fp + 'in-v-out_aqi_bins_'+pm_use+out_desc+'.png')
    plt.show()

######################################################################################
# Print/Save Additional Numbers for Paper
######################################################################################
uh_aqi_inds = np.where(pa_df_ss[opm_use]>=150.45)
print('n out PM > 150.45',len(uh_aqi_inds[0]))
print('unq monitors',len(np.unique(pa_df_ss['ID'].iloc[uh_aqi_inds])))
print('date out PM > 150.45',np.unique(pa_df_ss['time_loc'].iloc[uh_aqi_inds]))

# number of indoor PM at same AQI as outdoor above the moderate level
above_mod_inds = np.where(pa_df_ss[opm_use]>=35.45)[0]
tot_above_mod = len(above_mod_inds)
usi_atorab = len(np.where(pa_df_ss[ipm_use].iloc[usi_inds]>=35.45)[0])
u_atorab = len(np.where(pa_df_ss[ipm_use].iloc[u_inds]>=55.45)[0])
vh_atorab = len(np.where(pa_df_ss[ipm_use].iloc[vh_inds]>=150.45)[0])
print('total obs above mod level',tot_above_mod,'n in at or above out AQI',usi_atorab+u_atorab+vh_atorab)

# n ratio < 1 for bins above "good"
abv_good_inds = np.where(pa_df_ss[opm_use]>=12.05)
n_less1 = len(np.where(pa_df_ss['ratio'].iloc[abv_good_inds]<1.0)[0])
print('tot above good',len(abv_good_inds[0]),'n <1',n_less1)
                                   
# combine data into pandas array
row_names = ['si-good','si-moderate','si-usi','si-uh','si-vuh','si-hazd']
aqi_stats = pd.DataFrame(data={'stat':row_names,
                               'med_in':med_in,'lqt_in':lqt_in,'uqt_in':uqt_in,
                               'n_mon_in':n_mon_in,'med_ratio':med_ratio,'lqt_ratio':lqt_ratio,
                               'uqt_ratio':uqt_ratio})
aqi_stats.to_csv(out_file_fp+'aqi_stats_'+out_desc+'.csv')

# DONE with the area
