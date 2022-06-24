'''
boxplot_all_counties.py
   python script to load all counties we are working with and make box plot of
   PM on smoke-impacted and smoke-free days for 2020, write statistics from boxplots to a csv, 
   and calculate a linear regression of in v out PM across all area observations. 
written by: Katelyn O'Dell
05.03.21
'''
########################################################################################################
# user inputs
########################################################################################################
# directory with all the subfolders and data we need
scratch_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/'
# version of files to use
file_desc = 'fv' # fv = final version pre-submission
# make figures for main text or SI?
version = 'main' # or 'SI'

# specific files and paths
PA_data_fp = scratch_fp + 'ABmrg_clean_wsmoke/'
PAclean_metadata_fn = scratch_fp + 'ABmrg_clean/clean_stats_test_2020_nTRH_wUS_'+file_desc+'.csv'
PAall_metadata_fn = scratch_fp + 'ABmrg/PAmetadata_wprocessed_names_wUScoloc_d1000m_'+file_desc+'.csv'

# area strings we want included
areas_load = ['SLC','PNW','SF','LA','CFR']

# where to put out figures and files
out_fig_fp = '/home/kaodell/NSF/purple_air_smoke/Bonne_data/'
out_file_fp = '/home/kaodell/NSF/purple_air_smoke/Bonne_data/'
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
from scipy.stats import linregress, ks_2samp
import datetime as dt
import matplotlib.colors as colors

########################################################################################################
# load meta data
########################################################################################################
md_df_clean = pd.read_csv(PAclean_metadata_fn)
md_df_all = pd.read_csv(PAall_metadata_fn)

########################################################################################################
# load in/out monitors, combine data
########################################################################################################
# identify IDs to load
IDs_load = []
areas = []
for area in areas_load:
    inds = np.where(md_df_all['area_abbr']==area)[0]
    IDs_load = np.hstack([IDs_load,md_df_all['processed_names'].iloc[inds].values])
    areas = np.hstack([areas,[area]*len(inds)])
    
# load identified files and stack
count = 0
nodata = 0
nofile = 0
lons = []
lats = []
for IDind in range(len(IDs_load)):
    ID = IDs_load[IDind]
    if np.isnan(ID):
        nodata += 1
        continue
    fn_load = 'purpleair_PMAB20_mID'+str(int(ID)).zfill(4)+'_AnBmrg_wUScoloc_d1000m_clean_wsmoke_2020_'+file_desc+'_daily_.csv'
    if fn_load not in os.listdir(PA_data_fp):
        nofile += 1
        continue
    pa_df = pd.read_csv(PA_data_fp + fn_load)
    # add ID and area to the file
    pa_df['ID'] = [str(int(ID)).zfill(4)]*pa_df.shape[0]
    pa_df['area'] = [areas[IDind]]*pa_df.shape[0]
    # get lat/lon of these monitors
    ind = np.where(md_df_all['processed_names']==ID)[0]
    lons.append(md_df_all['in_lon'].iloc[ind].values[0])
    lats.append(md_df_all['in_lat'].iloc[ind].values[0])
    # stack monitors
    if count == 0:
        pa_df_all = pa_df
    else:
        pa_df_all = pd.concat([pa_df_all,pa_df])
    count += 1
print(count,'monitors available in range (note some may not have data)')

########################################################################################################
# plot on a map to make sure we got the right monitors
########################################################################################################
fig,ax = plt.subplots(ncols=1,nrows=1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='gray',alpha=0.5)
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'))
ax.coastlines('50m')
cs = ax.scatter(lons,lats,transform=ccrs.PlateCarree(),color='red')
plt.show()

########################################################################################################
# before making plots and calculating stats for paper,
# remove in and out nans, days with <50% of the time with observations, and get smoke inds
########################################################################################################
pa_df_all.reset_index(inplace=True,drop=True)
# identify where in or out is a nan and drop
naninds_in = np.where(np.isnan(pa_df_all['PM25_bj_in']))
naninds_out = np.where(np.isnan(pa_df_all['PM25_bj_out']))
naninds = np.unique(np.hstack([naninds_in,naninds_out]))
pa_df_nn1 = pa_df_all.drop(naninds)
pa_df_nn1.reset_index(inplace=True,drop=True)

# identify days with fewer than 50% observations and drop
ormv_inds = np.where(pa_df_nn1['pm_out_count']<(3*24.0))[0]
irmv_inds = np.where(pa_df_nn1['pm_in_count']<(3*24.0))[0]
rmv_inds = np.unique(np.hstack([ormv_inds,irmv_inds]))
pa_df_nn2 = pa_df_nn1.drop(rmv_inds)
pa_df_nn2.reset_index(inplace=True,drop=True)

# for si version, add smoke season flag and pull only summer and fall observations
if version == 'SI':
    ss_flag = []
    for i in range(pa_df_nn2.shape[0]):
        month = pa_df_nn2['time_loc'].iloc[i].split('-')[1]
        if month in ['06','07','08','09','10','11']:
            ss_flag.append(1)
        elif month in ['01','02','03','04','05','12']:
            ss_flag.append(0)
        else:
            print('month not recognized, datetime error')
            hi=bye
    pa_df_nn2['smoke_season_flag']=ss_flag
    # drop observatiosn outside smoke season
    nss_inds = np.where(pa_df_nn2['smoke_season_flag']==0)[0]
    pa_df_nn = pa_df_nn2.drop(nss_inds)
    pa_df_nn.reset_index(inplace=True,drop=True)
# for figure version in the main text, use all observations
elif version == 'main':
    pa_df_nn = pa_df_nn2

# idenfity smoke-impacted and smoke-free inds
smoke_inds = np.where(pa_df_nn['smoke_flag']==1)[0]
nsmoke_inds = np.where(pa_df_nn['HMS_flag']==0)[0]

# make negatives zeros - sometimes cf makes values negative, but we wait to change to zeros until after averaging
pa_df_nn['PM25_bj_in'] = np.where(pa_df_nn['PM25_bj_in']<0,0,pa_df_nn['PM25_bj_in'])
pa_df_nn['PM25_bj_out'] = np.where(pa_df_nn['PM25_bj_out']<0,0,pa_df_nn['PM25_bj_out'])

# also round to 2 decimal places, like in the original csv's
pa_df_nn['PM25_bj_in'] = np.round(pa_df_nn['PM25_bj_in'].values,2)
pa_df_nn['PM25_bj_out'] = np.round(pa_df_nn['PM25_bj_out'].values,2)    

########################################################################################################
# calculate linear regression of in vs out for paper for these areas - do not use in revised version
########################################################################################################
'''
stats_a = linregress(pa_df_nn['PM25_bj_out'].iloc[smoke_inds],pa_df_nn['PM25_bj_in'].iloc[smoke_inds])
# also do orthogonal least squares regression, linear ends up in paper, but we compare
from scipy import odr
def f(B, x):
    # Linear function y = m*x + b
    # B is a vector of the parameters, x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]
mydata = odr.Data(pa_df_nn['PM25_bj_out'].iloc[smoke_inds],pa_df_nn['PM25_bj_in'].iloc[smoke_inds])
linear = odr.Model(f)
myodr = odr.ODR(mydata,linear,beta0=[0.2,5.0])
stats_a_odr = myodr.run()

print('OLS out v in regression results', stats_a) # these are in the paper
print('Orthogonal LS out v in regression results', stats_a_odr.beta)

res = stats_a
x = pa_df_nn['PM25_bj_out'].iloc[smoke_inds]
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(x)-2)
print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
print(f"intercept (95%): {res.intercept:.6f}"
      f" +/- {ts*res.intercept_stderr:.6f}")
'''
########################################################################################################
# identify smoke and no smoke inds for each area for plotting and calculating stats
########################################################################################################
ba_inds = np.where(pa_df_nn['area']=='SF')[0]
la_inds = np.where(pa_df_nn['area']=='LA')[0]
sea_inds = np.where(pa_df_nn['area']=='PNW')[0]
slc_inds = np.where(pa_df_nn['area']=='SLC')[0]
cfr_inds = np.where(pa_df_nn['area']=='CFR')[0]

ba_ns_inds = ba_inds[np.where(pa_df_nn['HMS_flag'].iloc[ba_inds]==0)[0]]
la_ns_inds = la_inds[np.where(pa_df_nn['HMS_flag'].iloc[la_inds]==0)[0]]
sea_ns_inds = sea_inds[np.where(pa_df_nn['HMS_flag'].iloc[sea_inds]==0)[0]]
slc_ns_inds = slc_inds[np.where(pa_df_nn['HMS_flag'].iloc[slc_inds]==0)[0]]
cfr_ns_inds = cfr_inds[np.where(pa_df_nn['HMS_flag'].iloc[cfr_inds]==0)[0]]

ba_s_inds = ba_inds[np.where(pa_df_nn['smoke_flag'].iloc[ba_inds]==1)[0]]
la_s_inds = la_inds[np.where(pa_df_nn['smoke_flag'].iloc[la_inds]==1)[0]]
sea_s_inds = sea_inds[np.where(pa_df_nn['smoke_flag'].iloc[sea_inds]==1)[0]]
slc_s_inds = slc_inds[np.where(pa_df_nn['smoke_flag'].iloc[slc_inds]==1)[0]]
cfr_s_inds = cfr_inds[np.where(pa_df_nn['smoke_flag'].iloc[cfr_inds]==1)[0]]

########################################################################################################
# calc stats across monitors and write to file
########################################################################################################
area_names = ['SF','LA','PNW','SLC','CFR']
area_stats = pd.DataFrame(data={'area':area_names})
key_inds = [ [ba_inds,la_inds,sea_inds,slc_inds,cfr_inds],
             [ba_ns_inds,la_ns_inds,sea_ns_inds,slc_ns_inds,cfr_ns_inds],
             [ba_s_inds, la_s_inds, sea_s_inds, slc_s_inds,cfr_s_inds]]
keys = ['ad','ns','s'] # 'all days', 'no smoke', 'smoke'
for var in ['PM25_bj_in','PM25_bj_out']:
    for ki in range(3):
        key = keys[ki]
        key_med = []
        key_mean = []
        key_std = []
        key_n = []
        key_25pct = []
        key_75pct = []
        for i in range(5): 
            key_med.append(pa_df_nn[var].iloc[key_inds[ki][i]].median())
            key_mean.append(pa_df_nn[var].iloc[key_inds[ki][i]].mean())
            key_std.append(pa_df_nn[var].iloc[key_inds[ki][i]].std())
            key_n.append(len(key_inds[ki][i]))
            key_25pct.append(np.percentile(pa_df_nn[var].iloc[key_inds[ki][i]],25))
            key_75pct.append(np.percentile(pa_df_nn[var].iloc[key_inds[ki][i]],75))
        area_stats[var+key+' med']=key_med
        area_stats[var+key+' mean']=key_mean
        area_stats[var+key+' std']=key_std
        area_stats[var+key+' n']=key_n
        area_stats[var+key+' 25pct']=key_25pct
        area_stats[var+key+' 75pct']=key_75pct
area_stats.to_csv('/home/kaodell/NSF/purple_air_smoke/daily_area_stats_sflag_update_wUS_'+file_desc+'_'+version+'.csv')

# calculate where difference between SI and SF indoor is statistically significant
# using a ks test
all_smoke_inds = [ba_s_inds,la_s_inds,sea_s_inds,slc_s_inds,cfr_s_inds]
all_nonsmoke_inds = [ba_ns_inds,la_ns_inds,sea_ns_inds,slc_ns_inds,cfr_ns_inds]
i = 0
for name in area_names:
    smoke_inds_area = all_smoke_inds[i]
    nsmoke_inds_area = all_nonsmoke_inds[i]
    in_ks, in_pval = ks_2samp(pa_df_nn['PM25_bj_in'].iloc[smoke_inds_area], pa_df_nn['PM25_bj_in'].iloc[nsmoke_inds_area])
    print(name, 'in SI v SF, ks test',in_ks,in_pval)
    i += 1

# count times when indoor > 35 ug/m3 on all smoke days and heavily impacted smoke days
inds = np.where(pa_df_nn['PM25_bj_in'].iloc[smoke_inds]>=12.05)[0]
print('smoke-impacted indoor > 12',len(inds),len(inds)/len(smoke_inds))
inds = np.where(pa_df_nn['PM25_bj_in'].iloc[smoke_inds]>=35.45)[0]
print('smoke-impacted indoor > 35',len(inds),len(inds)/len(smoke_inds))
inds1 = smoke_inds[np.where(pa_df_nn['PM25_bj_out'].iloc[smoke_inds]>=55.45)[0]]
inds2 = np.where(pa_df_nn['PM25_bj_in'].iloc[inds1]>=35.45)[0]
print('heavily smoke-impacted indoor > 35:',len(inds2),len(inds2)/len(inds1))

# count times when in/out ratio is < 1 on smoke days when out >= 12 ug/m3
# also have code to do this in the initial_analysis_allPA code, but is faster to do it here
inds3 = smoke_inds[np.where(pa_df_nn['PM25_bj_out'].iloc[smoke_inds]>=12.05)]
inds4 = np.where((pa_df_nn['PM25_bj_in'].iloc[inds3]/pa_df_nn['PM25_bj_out'].iloc[inds3]) < 1)[0]
print('si ratio < 1, at or above moderate level',len(inds4),len(inds4)/len(inds3))

########################################################################################################
# make pdf of indoor concentrations on smoke-impacted days
########################################################################################################
data = pa_df_nn['PM25_bj_in'].iloc[smoke_inds]
data_hs = pa_df_nn['PM25_bj_in'].iloc[inds1]

# look at all smoke-impacted days and heavily smoke-impacted days
# getting data of the histogram
count, bins_count = np.histogram(data, bins=100)
count_hs, bins_count_hs = np.histogram(data_hs,bins=100)
# finding the PDF of the histogram using count values
pdf = count / sum(count)
pdf_hs = count_hs / sum(count_hs)
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
cdf_hs = np.cumsum(pdf_hs)

fig, ax = plt.subplots(1)
ax.plot(bins_count[1:],100.0*cdf,color='darkgray',label='All smoke-impacted days')
ax.plot(bins_count_hs[1:],100.0*cdf_hs,color='k',label='Heavily smoke-impacted days')
ax.plot([35.45,35.45],[0,100.1],'--',color='darkorange',label='24hr PM$_{2.5}$ standard')
ax.plot([12.05,12.05],[0,100.1],'--',color='gold',label='Annual PM$_{2.5}$ standard')
ax.set_ylim([0,100.1])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(loc='best',frameon=False)
ax.set_xlabel('Daily-mean Indoor PM$_{2.5}$ [$\mu$g m$^{-3}$]')
ax.set_ylabel('Percent of Observations [%]')
plt.savefig(out_fig_fp+'indoor_smokeimpacted_cdf_wUSregions_'+file_desc+'_'+version+'.png',dpi=300)
plt.show()

########################################################################################################
# make boxplots
########################################################################################################
# no smoke
fig, axs = plt.subplots(1,2,sharey=True,figsize=(12,5))
axs=axs.flatten()
bp1 = axs[0].boxplot([pa_df_nn['PM25_bj_in'].iloc[ba_ns_inds],
                      pa_df_nn['PM25_bj_in'].iloc[la_ns_inds],
                      pa_df_nn['PM25_bj_in'].iloc[sea_ns_inds],
                      pa_df_nn['PM25_bj_in'].iloc[slc_ns_inds],
                      pa_df_nn['PM25_bj_in'].iloc[cfr_ns_inds]],
                     labels=['   San Francisco\n   N:'+str(len(ba_ns_inds)),
                             '   Los Angeles\n   '+str(len(la_ns_inds)),
                             '   Seattle\n& Portland\n   '+str(len(sea_ns_inds)),
                             '   Salt Lake\n   '+str(len(slc_ns_inds)),
                             '   Denver\n   '+str(len(cfr_ns_inds))],
                     positions= [0.8,1.8,2.8,3.8,4.8],whis=(2.5,97.5),patch_artist=True,showfliers=True)
bp2 = axs[0].boxplot([pa_df_nn['PM25_bj_out'].iloc[ba_ns_inds],
                      pa_df_nn['PM25_bj_out'].iloc[la_ns_inds],
                      pa_df_nn['PM25_bj_out'].iloc[sea_ns_inds],
                      pa_df_nn['PM25_bj_out'].iloc[slc_ns_inds],
                      pa_df_nn['PM25_bj_out'].iloc[cfr_ns_inds]],
                     whis=(2.5,97.5),labels = ['','','','',''],
                     positions=[1.2,2.2,3.2,4.2,5.2],patch_artist=True,showfliers=True)
# smoke-impacted
bp3 = axs[1].boxplot([pa_df_nn['PM25_bj_in'].iloc[ba_s_inds],
                      pa_df_nn['PM25_bj_in'].iloc[la_s_inds],
                      pa_df_nn['PM25_bj_in'].iloc[sea_s_inds],
                      pa_df_nn['PM25_bj_in'].iloc[slc_s_inds],
                      pa_df_nn['PM25_bj_in'].iloc[cfr_s_inds]],
                     labels=['   San Francisco\n   N:'+str(len(ba_s_inds)),
                             '   Los Angeles\n   '+str(len(la_s_inds)),
                             '   Seattle\n& Portland\n   '+str(len(sea_s_inds)),
                             '   Salt Lake City\n   '+str(len(slc_s_inds)),
                             '   Denver\n   '+str(len(cfr_s_inds))],
                     positions= [0.8,1.8,2.8,3.8,4.8],whis=(2.5,97.5),patch_artist=True,showfliers=True)
bp4 = axs[1].boxplot([pa_df_nn['PM25_bj_out'].iloc[ba_s_inds],
                      pa_df_nn['PM25_bj_out'].iloc[la_s_inds],
                      pa_df_nn['PM25_bj_out'].iloc[sea_s_inds],
                      pa_df_nn['PM25_bj_out'].iloc[slc_s_inds],
                      pa_df_nn['PM25_bj_out'].iloc[cfr_s_inds]],labels=['','','','',''],
                     whis=(2.5,97.5), positions=[1.2,2.2,3.2,4.2,5.2],patch_artist=True,showfliers=True)
# format boxplot colors
area_colors = ['#984ea3','#ff7f00','#377eb8','#e41a1c','#4daf4a']
for bplot in (bp1,bp2,bp3,bp4):
    for i in range(5):
        bplot['medians'][i].set(color='darkgrey')
        bplot['caps'][2*i].set(color=area_colors[i])
        bplot['caps'][2*i+1].set(color=area_colors[i])
        bplot['whiskers'][2*i].set(color=area_colors[i])
        bplot['whiskers'][2*i+1].set(color=area_colors[i])
        bplot['fliers'][i].set(markeredgecolor='darkgrey',
                               markersize=2)
        # make indoor darker / outdoor lighter
        if bplot in [bp1,bp3]:
            bplot['boxes'][i].set(color=area_colors[i])
        else:
            bplot['boxes'][i].set(color=area_colors[i],alpha=0.3)
# set scales
for i in range(2):
    axs[i].set_yscale('log')
    axs[i].tick_params(axis='x',labelsize=10)
    axs[i].spines["right"].set_visible(False)
    axs[i].spines["top"].set_visible(False)
# set titles, labels, and axes limits
axs[0].set_title('Smoke-Free',fontsize=12)
axs[1].set_title('Smoke-Impacted',fontsize=12)
axs[0].set_ylabel('Daily PM$_{2.5}$ [$\mu$g m$^{-3}$]')
axs[0].set_ylim(0.5,240)
axs[1].set_ylim(0.5,240)
fig.show()
plt.savefig(out_fig_fp+'boxplot_smoke_split_allcounties_daily_wUS_'+file_desc+'_'+version+'.png',dpi=300)
