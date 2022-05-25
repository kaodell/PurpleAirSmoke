'''
calc_inout_ratio.py
    python script to calculate the ratio of indoor to outdoor PM for co-located purple air monitors
    and save to file to use on local machine to make figures
written by: Katelyn O'Dell
v0 - 04.12.21 - initial script
'''
###########################################################################################
# user inputs
###########################################################################################
# version of files we are using
file_desc = 'fv' # fv = final version pre-submission

# make data for main text version of figures or SI versions?
version = 'SI' # or 'SI'

# location of PA processed files and metadata
PA_data_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/ABmrg_clean_wsmoke/'
PAall_metadata_fn ='/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/ABmrg/PAmetadata_wprocessed_names_wUScoloc_d1000m_'+file_desc+'.csv'

# which PM cf to use
iPM_use = 'PM25_bj_in'
oPM_use = 'PM25_bj_out'

# where to put out figures
out_fig_fp = '/home/kaodell/NSF/purple_air_smoke/Bonne_data/figures/'
out_file_fp = '/home/kaodell/NSF/purple_air_smoke/Bonne_data/'

###########################################################################################
# load modules
###########################################################################################
import pandas as pd
import numpy as np
import scipy.stats as st
import os

###########################################################################################
# load data
###########################################################################################
# load metadata
md_df_all = pd.read_csv(PAall_metadata_fn,dtype={'in_name':'string','in_AID':int,
                                   'in_AID_tp':int,'in_lat':float,'in_lon':float,
                                   'out_name':'string','out_AID':int,'out_AID_tp':int,
                                   'out_lat':float,'out_lon':float,'distance':float,'in_AID_tp_key':'string','out_AID_tp_key':'string','in_mon_type':'string',
                                   'out_mon_type':'string','in_BID':int,
                                   'out_BID_tp':int,'in_BID_tp':int,'in_BID_tp_key':'string','out_BID_tp_key':'string','county_names':'string','FIPS_code':float,'area_abbr':'string','processed_names':float})

# identify IDs to load
IDs_load1 = md_df_all['processed_names'].values
rmv_inds = np.where(np.isnan(IDs_load1))
IDs_load = np.delete(IDs_load1,rmv_inds)

################################################################################################
# pre-allocate arrays of info we want to save for each monitor
################################################################################################
IDs, lats, lons, names, inA_tpID = [],[],[],[],[]

# PM stats
oPM_mean_all = []
inPM_mean_si, inPM_mean_ns, oPM_mean_si, oPM_mean_ns = [], [], [], []
inPM_median_si, inPM_median_ns, oPM_median_si, oPM_median_ns = [], [], [], []

# ratio stats for all days, smoke days, and no-smoke days
ratio_mean, ratio_median, ratio_25pct, ratio_75pct = [], [], [], []
ratio_mean_s, ratio_median_s, ratio_25pct_s, ratio_75pct_s = [], [], [], []
ratio_mean_ns, ratio_median_ns, ratio_25pct_ns, ratio_75pct_ns = [], [], [], []

# slopes for in v out
smoke_slope, smoke_slopes_pval = [], []
nosmoke_slope, nosmoke_slopes_pval = [], []

# correlation of monitors
corrs_all, corrs_s, corrs_ns, ks_iPM_pval, ks_ratio_pval = [], [], [], [], []

# number of observations
num_all, num_s, num_ns, num_all_ratio, num_s_ratio, num_ns_ratio = [] ,[], [], [], [], []
n_count_rmv = []
total_obs_b4 = []

################################################################################################
# start looping through files
################################################################################################
nofile=0
print('loading',len(IDs_load),'IDs')
for ID in IDs_load:
    ################################################################################################
    # in loop: load files and clean
    ################################################################################################
    fn_load = 'purpleair_PMAB20_mID'+str(int(ID)).zfill(4)+'_AnBmrg_wUScoloc_d1000m_clean_wsmoke_2020_'+file_desc+'_daily_.csv'
    if fn_load not in os.listdir(PA_data_fp):
        nofile += 1
        continue
    print(ID)
    pa_df_all1 = pd.read_csv(PA_data_fp + fn_load,dtype={'time_loc':'string','PM25_bj_in':float,'PM25_bj_out':float,'PM25_ds_in':float,'PM25_ds_out':float,'PM25_LRAPA_in':float,'PM25_LRAPA_out':float,'HMS_flag':int,'event_len':int,'pm_in_count':int,'pm_out_count':int,'RH_in':float,'RH_out':float,'T_in':float,'T_out':float,'RH_in_count':int,'RH_out_count':int,'T_in_count':int,'T_out_count':int,'smoke_flag':int})
    pa_df_all1.reset_index(inplace=True,drop=True)
    
    # remove inds where either in or out is a nan
    in_nan_inds = np.where(np.isnan(pa_df_all1[iPM_use]))
    out_nan_inds = np.where(np.isnan(pa_df_all1[oPM_use]))
    nan_inds = np.unique(np.hstack([in_nan_inds,out_nan_inds])[0])
    pa_df_all2 = pa_df_all1.drop(nan_inds)
    pa_df_all2.reset_index(inplace=True,drop=True)
    if pa_df_all2.shape[0]==0:
        print('file all nans, moving on')
        continue
    
    # remove %inds with <50% of the day available
    inds_o_rmv = np.where(pa_df_all2['pm_out_count']<0.5*(6*24))[0]
    inds_i_rmv = np.where(pa_df_all2['pm_in_count']<0.5*(6*24))[0]
    inds_rmv = np.unique(np.hstack([inds_o_rmv,inds_i_rmv]))
    pa_df3 = pa_df_all2.drop(inds_rmv)
    pa_df3.reset_index(inplace=True,drop=True)
    # save n obs removed, n obs before below next section
    
    ################################################################################################
    # in loop: determine if we are calculating stats for main version or si version figures
    ################################################################################################
    if version == 'SI':
        # add smoke season flag and pull only summer and fall observations
        ss_flag = []
        for i in range(pa_df3.shape[0]):
            month = pa_df3['time_loc'].iloc[i].split('-')[1]
            if month in ['06','07','08','09','10','11']:
                ss_flag.append(1)
            elif month in ['01','02','03','04','05','12']:
                ss_flag.append(0)
            else:
                print('month not recognized, timestamp error')
                hi=bye
        pa_df3['smoke_season_flag']=ss_flag
        nss_inds = np.where(pa_df3['smoke_season_flag']==0)[0]
        pa_df = pa_df3.drop(nss_inds)
        pa_df.reset_index(inplace=True,drop=True)
    elif version == 'main':
        pa_df = pa_df3.copy()
    else:
        print('version selection error')
        hi=bye

    if pa_df.shape[0]==0:
        'no data in selected smoke season'
        continue

    ################################################################################################
    # in loop: calculate total numbers, identify smoke inds, nonsmoke inds and relevant metadata
    ################################################################################################
    # now save cleaning numbers from above
    n_count_rmv.append(len(inds_rmv)) # days with less than 50% avail obs for average
    total_obs_b4.append(pa_df_all2.shape[0]) # observation count after removing in and out nans
    
    # count total obs for analysis and get si and sf inds
    num_all.append(pa_df.shape[0])
    s_inds = np.where(pa_df['smoke_flag']==1)
    num_s.append(len(s_inds[0]))
    ns_inds = np.where(pa_df['HMS_flag']==0)
    num_ns.append(len(ns_inds[0]))

    # make negative PM 0
    pa_df[oPM_use]=np.where(pa_df[oPM_use]<0,0,pa_df[oPM_use])
    pa_df[iPM_use]=np.where(pa_df[iPM_use]<0,0,pa_df[iPM_use])
    
    # round to 0.01 decimal, reported by PA, and add ratio to dataframe
    out_for_ratio = np.around(pa_df[oPM_use].values,decimals=2)
    in_for_ratio = np.around(pa_df[iPM_use].values,decimals=2)
    ratio = in_for_ratio/out_for_ratio
    pa_df['ratio'] = ratio

    # count non-nan ratio obs for each group as well
    num_s_ratio.append(len(np.where(np.isfinite(pa_df['ratio'].iloc[s_inds]))[0]))
    num_ns_ratio.append(len(np.where(np.isfinite(pa_df['ratio'].iloc[ns_inds]))[0]))
    num_all_ratio.append(len(np.where(np.isfinite(pa_df['ratio']))[0]))
    
    # get lat/lon for groups for plotting
    md_ind = np.where(float(ID)==md_df_all['processed_names'])
    lats.append(md_df_all['in_lat'].iloc[md_ind].values[0])
    lons.append(md_df_all['in_lon'].iloc[md_ind].values[0])
    names.append(md_df_all['in_name'].iloc[md_ind].values[0])
    inA_tpID.append(md_df_all['in_AID_tp'].iloc[md_ind].values[0])

    ################################################################################################
    # in loop: calculate ratio stats (there are more succint ways to do this, but this works)
    ################################################################################################   
    # ratio stats all days
    ratio_mean.append(np.nanmean(pa_df['ratio']))
    ratio_median.append(np.nanmedian(pa_df['ratio']))
    ratio_25pct.append(np.nanpercentile(pa_df['ratio'].values,25))
    ratio_75pct.append(np.nanpercentile(pa_df['ratio'].values,75))

    # ratio for smoke impacted and smoke free days
    ratio_mean_ns.append(np.nanmean(pa_df['ratio'].iloc[ns_inds].values))
    ratio_mean_s.append(np.nanmean(pa_df['ratio'].iloc[s_inds].values))
    ratio_median_ns.append(np.nanmedian(pa_df['ratio'].iloc[ns_inds].values))
    ratio_median_s.append(np.nanmedian(pa_df['ratio'].iloc[s_inds].values))
    ratio_25pct_s.append(np.nanpercentile(pa_df['ratio'].iloc[s_inds].values,25))
    ratio_75pct_s.append(np.nanpercentile(pa_df['ratio'].iloc[s_inds].values,75))
    ratio_25pct_ns.append(np.nanpercentile(pa_df['ratio'].iloc[ns_inds].values,25))
    ratio_75pct_ns.append(np.nanpercentile(pa_df['ratio'].iloc[ns_inds].values,75))

    # mean and median indoor PM on smoke-impacted and smoke-free days
    inPM_mean_si.append(np.nanmean(pa_df[iPM_use].iloc[s_inds].values))
    inPM_mean_ns.append(np.nanmean(pa_df[iPM_use].iloc[ns_inds].values))
    inPM_median_si.append(np.nanmedian(pa_df[iPM_use].iloc[s_inds].values))
    inPM_median_ns.append(np.nanmedian(pa_df[iPM_use].iloc[ns_inds].values))
    
    # mean and median outdoor PM on smoke-impacted and smoke-free days
    oPM_mean_si.append(np.nanmean(pa_df[oPM_use].iloc[s_inds].values))
    oPM_mean_ns.append(np.nanmean(pa_df[oPM_use].iloc[ns_inds].values))
    oPM_median_si.append(np.nanmedian(pa_df[oPM_use].iloc[s_inds].values))
    oPM_median_ns.append(np.nanmedian(pa_df[oPM_use].iloc[ns_inds].values))
    # all-day outdoor PM mean
    oPM_mean_all.append(pa_df[oPM_use].mean())

    # calculate linear regression of indoor and outdoor PM on smoke days
    if len(s_inds[0])==0:
        smoke_slope.append(np.nan)
        smoke_slopes_pval.append(np.nan)
    else:
        slope, intercept, r, p, se = st.linregress(pa_df[oPM_use].iloc[s_inds].values, pa_df[iPM_use].iloc[s_inds].values)
        smoke_slope.append(slope)
        smoke_slopes_pval.append(p)
        
    if len(ns_inds[0])==0:
        nosmoke_slope.append(np.nan)
        nosmoke_slopes_pval.append(np.nan)
    else:
        slope, intercept, r, p, se = st.linregress(pa_df[oPM_use].iloc[ns_inds].values, pa_df[iPM_use].iloc[ns_inds].values)
        nosmoke_slope.append(slope)
        nosmoke_slopes_pval.append(p)
    
    ################################################################################################
    # in loop: calculate correlation
    ################################################################################################  
    # confirm there is enough data
    if num_all[-1]<3:
        corr_all = np.nan
        corr_ns = np.nan
        corr_s = np.nan
        ks_iPM_pval.append(np.nan)
        ks_ratio_pval.append(np.nan)
    elif np.any([num_s[-1]<3,num_ns[-1]<3]):
        corr_all = st.spearmanr(pa_df[oPM_use],pa_df[iPM_use],nan_policy='omit')[0] # no nans, but good to define
        corr_ns = np.nan
        corr_s = np.nan
        ks_iPM_pval.append(np.nan)
        ks_ratio_pval.append(np.nan)
    else:
        corr_all = st.spearmanr(pa_df[oPM_use],pa_df[iPM_use],nan_policy='omit')[0]
        corr_ns = st.spearmanr(pa_df[oPM_use].iloc[ns_inds],pa_df[iPM_use].iloc[ns_inds],nan_policy='omit')[0]
        corr_s = st.spearmanr(pa_df[oPM_use].iloc[s_inds],pa_df[iPM_use].iloc[s_inds],nan_policy='omit')[0]

        # Is in PM statistically different on smoke days?
        # do K-S test for non-parametric... we don't use this ultimately in the paper
        ks_iPM_pval.append(st.ks_2samp(pa_df[iPM_use].iloc[s_inds],pa_df[iPM_use].iloc[ns_inds],alternative='two-sided')[1])
        ks_ratio_pval.append(st.ks_2samp(pa_df['ratio'].iloc[s_inds],pa_df['ratio'].iloc[ns_inds],alternative='two-sided')[1])
    corrs_all.append(corr_all)
    corrs_ns.append(corr_ns)
    corrs_s.append(corr_s)   
    IDs.append(ID)

# write this data to file to download and plot on local machine in code 'plot_PAratio_stats.py'
overall_stats_df = pd.DataFrame(data={'ID':IDs,'lon':lons,'lat':lats,
                                      'names':names,'in_AID_tp':inA_tpID,
                                      # PM stats
                                      'inPM_mean_si':inPM_mean_si,'inPM_mean_ns':inPM_mean_ns,
                                      'oPM_mean_si':oPM_mean_si,'oPM_mean_ns':oPM_mean_ns,
                                      'inPM_median_si':inPM_median_si,'inPM_median_ns':inPM_median_ns,
                                      'oPM_median_si':oPM_median_si,'oPM_median_ns':oPM_median_ns,
                                      # ratio stats
                                      'ratio_median_ad':ratio_median,'ratio_median_s':ratio_median_s,'ratio_median_ns':ratio_median_ns,
                                      'ratio_mean_ad':ratio_mean,'ratio_mean_s':ratio_mean_s,'ratio_mean_ns':ratio_mean_ns,
                                      'ratio_25pct_s':ratio_25pct_s,'ratio_75pct_s':ratio_75pct_s,
                                      'ratio_25pct_ns':ratio_25pct_ns,'ratio_75pct_ns':ratio_75pct_ns,
                                      'ratio_25pct_ad':ratio_25pct,'ratio_75pct_ad':ratio_75pct,
                                      # slopes
                                      'smoke_slopes':smoke_slope, 'nosmoke_slopes':nosmoke_slope,
                                      'smoke_slopes_pval':smoke_slopes_pval, 'nosmoke_slopes_pval':nosmoke_slopes_pval,
                                      # correlations
                                      'spearman_r_ad':corrs_all,'spearman_r_s':corrs_s,'spearman_r_ns':corrs_ns,
                                      'ks_iPM_pval':ks_iPM_pval,'ks_ratio_pval':ks_ratio_pval,
                                      # number of observations
                                      'num_s':num_s,'num_ns':num_ns,'num_all':num_all,
                                      'num_s_ratio':num_s_ratio,'num_ns_ratio':num_ns_ratio,'num_all_ratio':num_all_ratio,
                                      'n_count_rmv':n_count_rmv,
                                      'total_obs_b4':total_obs_b4})
overall_stats_df.to_csv(out_file_fp + 'overall_mon_stats_'+file_desc+'_'+version+'_revisions.csv')
