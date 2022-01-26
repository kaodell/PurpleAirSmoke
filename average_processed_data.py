'''
average_processed_data.py
python script to create daily and hourly merges of the purple air sensors
v0 - initial code, 04.03.21 (getting my COVID vaccine today!!!)
written by Katelyn O'Dell
'''
#####################################################################
# user inputs
#####################################################################
file_desc = 'fv' # fv = final version
scratch_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/' # folder on the scratch directory with the files
PA_fp = scratch_fp + 'ABmrg_clean_wsmoke/' # folder with purple air files to load
out_fp = scratch_fp + 'ABmrg_clean_wsmoke/' # where to put out files
metadata_fn = scratch_fp + 'ABmrg/PAmetadata_wprocessed_names_wUScoloc_d1000m_'+file_desc+'.csv' # file metadata

#####################################################################
# load modules
#####################################################################
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os

##################################################################
# load metadata
##################################################################
all_md_df = pd.read_csv(metadata_fn,dtype={'in_name':'string','in_AID':int,
                                   'in_AID_tp':int,'in_lat':float,'in_lon':float,
                                   'out_name':'string','out_AID':int,'out_AID_tp':int,
                                   'out_lat':float,'out_lon':float,'distance':float,'in_AID_tp_key':'string','out_AID_tp_key':'string','in_mon_type':'string',
                                   'out_mon_type':'string','in_BID':int,
                                   'out_BID_tp':int,'in_BID_tp':int,'in_BID_tp_key':'string','out_BID_tp_key':'string','county_names':'string','FIPS_code':float,'area_abbr':'string','processed_names':float})

#####################################################################
# load data and calculate averages
#####################################################################
for f_ID in all_md_df['processed_names']:
    if pd.isna(f_ID):
        continue
    # look for filename in the inout merge folder
    fn = 'purpleair_PMAB20_mID'+str(int(f_ID)).zfill(4)+'_AnBmrg_wUScoloc_d1000m_clean_wsmoke_2020_'+file_desc+'.csv'
    if fn not in os.listdir(PA_fp):
        # no file available (data removed in cleaning or in-out pairing)
        continue
    # load file
    pa_df = pd.read_csv(PA_fp +fn,dtype={'time':'string','PM25_bj_in':float,'PM25_ds_in':float,'PM25_LRAPA_in':float,'PMcf_1_avg_in':float,'PMatm_in':float,'T_in':float,'RH_in':float,'PM25_bj_out':float,'PM25_ds_out':float,'PM25_LRAPA_out':float,'PMcf1_avg_out':float,'PMatm_out':float,'T_out':float,'RH_out':float,'HMS_flag':int,'event_len':int})
    
    # note: barkjohn values can go negative (low PM and high RH), but leave these in before averaging
    # this will maintain linearity of correction, ie values are the same if we correct now vs post-averaging

    # only barkjohn and LRAPA should ever go negative, double check before averaging
    for var_str in ['PM25_ds','PMatm','PMcf_1_avg']:
        if pa_df[var_str+'_in'].min() < 0:
            print('negative error',var_str,'in')
            hi=bye
        elif pa_df[var_str+'_out'].min() < 0:
            print('negative error',var_str,'out')
            hi=bye

    # also note: we only downloaded 2020 PA data - but this is by UTC time, so we are missing the last
    # 7-8 hours of the local time year here. Since many monitors don't go to the end of the year, we focus on smoke
    # and using only HMS flag vs HMS + PM avg flag doesn't change our main conclusions, this is OK
    
    # create one datetime without minutes and seconds, and one datetime without hours to facilitate averaging
    th_dts_loc = []
    td_dts = []
    for t in pa_df['datetime_loc'].values:
        # cut off minutes, seconds and tz info
        th_loc = t[:-12]
        th_dts_loc.append(th_loc)
        # cut off hours, minutes, seconds, and tz ifno
        td = t[:-15]
        td_dts.append(td)
    # create hourly groups
    pa_df['hour_loc'] = th_dts_loc
    hourly_groups = pa_df.groupby(th_dts_loc)
    time_loc_hourly = hourly_groups['hour_loc'].first()
    # average across hourly groups
    pa_df_hourly_values = hourly_groups.mean()
    pa_df_hourly_counts = hourly_groups.count() # doesn't count nan values
    # create daily groups and average
    pa_df_daily_values = pa_df.groupby(td_dts).mean()
    pa_df_daily_counts = pa_df.groupby(td_dts).count() # again, doesn't count nans
    
    # create new dataframe with the variables we want from above
    pa_df_hourly = pd.DataFrame(data = {'time_loc':time_loc_hourly,
                                        # pm and smoke averages
                                        'PM25_bj_in':pa_df_hourly_values['PM25_bj_in'].values,
                                        'PM25_bj_out':pa_df_hourly_values['PM25_bj_out'].values,
                                        'PM25_ds_in':pa_df_hourly_values['PM25_ds_in'].values,
                                        'PM25_ds_out':pa_df_hourly_values['PM25_ds_out'].values,
                                        'PM25_LRAPA_in':pa_df_hourly_values['PM25_LRAPA_in'].values,
                                        'PM25_LRAPA_out':pa_df_hourly_values['PM25_LRAPA_out'].values,
                                        'PMatm_in':pa_df_hourly_values['PMatm_in'].values,
                                        'PMatm_out':pa_df_hourly_values['PMatm_out'].values,
                                        'PMcf_1_avg_in':pa_df_hourly_values['PMcf_1_avg_in'].values,
                                        'PMcf_1_avg_out':pa_df_hourly_values['PMcf_1_avg_out'].values,
                                        'HMS_flag':pa_df_hourly_values['HMS_flag'].values,
                                        'event_len':pa_df_hourly_values['event_len'].values,
                                        # pm counts
                                        'pm_in_count':pa_df_hourly_counts['PM25_bj_in'].values,
                                        'pm_out_count':pa_df_hourly_counts['PM25_bj_out'].values,
                                        # met averages
                                        'RH_in':pa_df_hourly_values['RH_in'],
                                        'RH_out':pa_df_hourly_values['RH_out'],
                                        'T_in':pa_df_hourly_values['T_in'],
                                        'T_out':pa_df_hourly_values['T_out'],
                                        # met counts
                                        'RH_in_count':pa_df_hourly_counts['RH_in'],
                                        'RH_out_count':pa_df_hourly_counts['RH_out'],
                                        'T_in_count':pa_df_hourly_counts['T_in'],
                                        'T_out_count':pa_df_hourly_counts['T_out'],})

    pa_df_daily = pd.DataFrame(data = {'time_loc':pa_df_daily_values.index,
                                       # pm and smoke averages
                                        'PM25_bj_in':pa_df_daily_values['PM25_bj_in'].values,
                                        'PM25_bj_out':pa_df_daily_values['PM25_bj_out'].values,
                                       'PM25_ds_in':pa_df_daily_values['PM25_ds_in'].values,
                                        'PM25_ds_out':pa_df_daily_values['PM25_ds_out'].values,
                                       'PM25_LRAPA_in':pa_df_daily_values['PM25_LRAPA_in'].values,
                                        'PM25_LRAPA_out':pa_df_daily_values['PM25_LRAPA_out'].values,
                                       'PMatm_in':pa_df_daily_values['PMatm_in'].values,
                                       'PMatm_out':pa_df_daily_values['PMatm_out'].values,
                                       'PMcf_1_avg_in':pa_df_daily_values['PMcf_1_avg_in'].values,
                                       'PMcf_1_avg_out':pa_df_daily_values['PMcf_1_avg_out'].values,
                                        'HMS_flag':pa_df_daily_values['HMS_flag'].values,
                                        'event_len':pa_df_daily_values['event_len'].values,
                                       # pm counts
                                        'pm_in_count':pa_df_daily_counts['PM25_bj_in'].values,
                                        'pm_out_count':pa_df_daily_counts['PM25_bj_out'].values,
                                       # met averages
                                       'RH_in':pa_df_daily_values['RH_in'],
                                        'RH_out':pa_df_daily_values['RH_out'],
                                        'T_in':pa_df_daily_values['T_in'],
                                        'T_out':pa_df_daily_values['T_out'],
                                       # met counts
                                        'RH_in_count':pa_df_daily_counts['RH_in'],
                                        'RH_out_count':pa_df_daily_counts['RH_out'],
                                        'T_in_count':pa_df_daily_counts['T_in'],
                                        'T_out_count':pa_df_daily_counts['T_out'],})
    pa_df_hourly.reset_index(inplace=True,drop=True)
    pa_df_daily.reset_index(inplace=True,drop=True)
    
    # check for too many values in the average, remember sometimes there are duplicate t obs, but shouldn't be above 7
    if pa_df_hourly['pm_in_count'].max()>7:
        print('!!! error, hourly (unless DST)!!!') # will be 12 during DST switch in fall
        print(pa_df_hourly['time_loc'].iloc[np.where(pa_df_hourly['pm_in_count']>6)].values)
        print(pa_df_hourly['pm_in_count'].iloc[np.where(pa_df_hourly['pm_in_count']>6)].values)
        
    if pa_df_daily['pm_in_count'].max()>150: # 150 for 'fall back' time change
        print('!!! error, daily (unless DST) !!!')
        
#####################################################################
# still in loop: update smoke flag with PM flag for daily and hourly
#####################################################################
    # add smoke flag based on daily PM averages
    # identify annual non-smoke PM average at monitor
    nHMS_inds = np.where(pa_df_daily['HMS_flag']==0)
    PM_dmean = np.nanmean(pa_df_daily['PM25_bj_out'].iloc[nHMS_inds])
    PM_dstd = 1.5*(np.nanstd(pa_df_daily['PM25_bj_out'].iloc[nHMS_inds]))
    # find elevated days PM > non-smoke mean + 1 sigma
    delv_inds = np.where(pa_df_daily['PM25_bj_out'].values>(PM_dmean+PM_dstd))[0]
    # find elevated days that are also HMS days, these are our new smoke inds
    dsi_inds = delv_inds[np.where(pa_df_daily['HMS_flag'].iloc[delv_inds]==1)[0]]
    dsmoke_flag = np.zeros(pa_df_daily.shape[0])
    dsmoke_flag[dsi_inds]=1
    pa_df_daily['smoke_flag']=dsmoke_flag

    # find hourly inds where there is a smoke day, based on new smoke flag above
    hsmoke_flag = []
    for dhi in range(pa_df_hourly.shape[0]):
        day = pa_df_hourly['time_loc'].iloc[dhi][:10] # get day 
        dind = np.where(pa_df_daily['time_loc']==day)[0][0] # find inds for that day in the daily file
        hsmoke_flag.append(dsmoke_flag[dind])
    pa_df_hourly['smoke_flag']=hsmoke_flag

    # print PM stats
    print('daily stats')
    print('mean:',PM_dmean)
    print('1.5 x std:',PM_dstd)
    # print('obs (incl nans):',len(PM_dmean))
    
#####################################################################
# still in loop: write to file
#####################################################################
    pa_df_hourly.to_csv(out_fp + fn[:-4] + '_hourly.csv')
    print('hourly ',f_ID,' saved')
    pa_df_daily.to_csv(out_fp + fn[:-4] + '_daily_.csv')
    print('daily ',f_ID, ' saved')
    print(np.unique(pa_df_daily['HMS_flag'].values))

#####################################################################
# plot daily, hourly, with loaded 10-minute to check
#####################################################################
    '''
    fig, ax = plt.subplots(2,1)
    axs = ax.flatten()
    # make datetime local without tzinfo, plotting shifts all timezones to UTC 
    tplot = []
    for t in pa_df['datetime_loc'].values:
        tplot.append(t[:-6])
    datetime_10m = pd.to_datetime(tplot)#pa_df['datetime_loc'].values)
    datetime_1h = pd.to_datetime(pa_df_hourly['time_loc'].values)
    datetime_1d = pd.to_datetime(pa_df_daily['time_loc'].values)
    axs[0].scatter(datetime_10m[:10000],pa_df['RH_in'].iloc[:10000],s=1,label='10m')
    axs[0].scatter(datetime_1h[:1000],pa_df_hourly['RH_in'].iloc[:1000],s=1,label='hourly')
    axs[0].scatter(datetime_1d[:42],pa_df_daily['RH_in'].iloc[:42],s=1,label='daily')
    axs[1].scatter(datetime_10m[:10000],pa_df['RH_out'].iloc[:10000],s=1)
    axs[1].scatter(datetime_1h[:1000],pa_df_hourly['RH_out'].iloc[:1000],s=1)
    axs[1].scatter(datetime_1d[:42],pa_df_daily['RH_out'].iloc[:42],s=1)
    axs[1].set_xlabel('time')
    axs[0].set_ylabel('in RH')
    axs[1].set_ylabel('out RH')
    fig.legend()
    fig.show()
    #hi=bye
    '''
