'''
AB_clean_stats.py
python3 script to load AB merge, clean, calculate stats, and apply correction factor.
v0 - initial code, 02.14.2021 (happy valentiles day!)
v1 - edited to work with all purple air monitors, 03.13.21 - snow day!
v2 - name changed to AB_clean_stats.py and only uses data from 2020

written by Katelyn O'Dell
'''
################################################################################
# User Inputs
################################################################################
file_desc = 'fv'
metadata_fn ='/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/ABmrg/PAmetadata_wprocessed_names_wUScoloc_d1000m_'+file_desc+'.csv'
mrg_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/ABmrg/'
out_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/ABmrg_clean/'
out_fp_inout = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/inout_mrg_clean/'
out_fig_fp = '/home/kaodell/NSF/purple_air_smoke/figures/'

################################################################################
# Load Modules
################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
from timezonefinder import TimezoneFinder
import datetime as dt
import pytz

################################################################################
# Pre-Allocate Arrays and Load Metadata
################################################################################
# pre-allocate arrays
b4stats_all = [-999.9,-999.9,-999.9,-999.9,-999.9]
after_stats_all = [-999.9,-999.9,-999.9,-999.9,-999.9]
nTArm = [] # observations removed based on temp in A file
nRHArm = [] # observations removed based on RH in A file
nTBrm = [] # was for testing temp in B file - we ultimately don't use it
nRHBrm = [] # was for testing RH in B file - we ultimately don't use it
nPMrm = [] # observations removed based on PM A-B sensor agreement
nPMhigh_rm = [] # observations removed based on PM observations outside of sensor range
start_time = [] # first observation time
stop_time = [] # last observation time
MAE_b4 = [] # A-B mean absolute error b4 cleaning
MAE_after = [] # A-B mean absolute error after cleaning
nAnans = [] # number nans in A PM sensor obs
nBnans = [] # number nans in B PM sensor obs
mis_clean = [] # IDs for cleaned files
flag = [] # flag indicating cleaning method based on monitor type (PA-I vs PA-II)
PM_means = [] # mean PM for the monitor
nobs_b4 = [] # number of observations before cleaning
nobs_after = [] # number of observations after cleaning
npairs = [] # number of paried observations after cleaning

# load metadata
metadata_df = pd.read_csv(metadata_fn,dtype={'in_name':'string','in_AID':int,
                                   'in_AID_tp':int,'in_lat':float,'in_lon':float,
                                   'out_name':'string','out_AID':int,'out_AID_tp':int,
                                   'out_lat':float,'out_lon':float,'distance':float,'in_AID_tp_key':'string','out_AID_tp_key':'string','in_mon_type':'string',
                                   'out_mon_type':'string','in_BID':int,
                                   'out_BID_tp':int,'in_BID_tp':int,'in_BID_tp_key':'string','out_BID_tp_key':'string','county_names':'string','FIPS_code':float,'area_abbr':'string','processed_names':float})

################################################################################
# Loop through sites and load data
################################################################################
#for mi in range(13,40): # for testing
for mi in range(metadata_df.shape[0]):
    snum_n = metadata_df['processed_names'].iloc[mi]
    slon = metadata_df['in_lon'].iloc[mi]
    slat = metadata_df['in_lat'].iloc[mi]
    if np.isnan(snum_n):
        print(mi,'no files')
        continue
    empty = False # change to true for empty cleaned files, so we don't process 'out' if 'in' is empty anyway
    snum = str(int(snum_n)).zfill(4)
    # loop through and load inside and outside
    for inout_flag in ['in','out']:
        if empty:
            continue
        loadfn = 'purpleair_colocmID'+snum+'_mrg_wUScoloc_d1000m_'+file_desc+'_'+inout_flag+'.csv'
        mrg = pd.read_csv(mrg_fp + loadfn,
                          dtype={'time':'string','PMA_atm':float,'PMA_cf1':float,
                                 'RH_A':float,'T_A':float,'atime':'string',
                                 'PMB_atm':float,'PMB_cf1':float,'RH_B':float,
                                 'T_B':float,'btime':'string'})
        
        ################################################################################
        # calc stats, clean data, clac stats
        ################################################################################
        # pull just 2020 data
        # identify local timezone
        tzname = TimezoneFinder().timezone_at(lng=slon,lat=slat)
        mrg_datetime = pd.to_datetime(mrg['time']) # this is in UTC
        datetime_locs = []
        years = []
        for datetime in mrg_datetime:
            datetime_utc = datetime.replace(tzinfo=pytz.utc)
            datetime_loc = datetime_utc.astimezone(pytz.timezone(tzname))
            datetime_locs.append(datetime_loc)
            years.append(datetime_loc.year)
        mrg['year'] = np.array(years)

        mrg = mrg[mrg.year == 2020]
        mrg.reset_index(inplace=True,drop=True)

        if len(mrg)<1:
            print('no data for selected year, skipping')
            empty=True
            continue
        
        # remove places where A or B is a nan
        Ani = np.where(np.isnan(mrg['PMA_atm']))
        Bni = np.where(np.isnan(mrg['PMB_atm']))
        ABni = np.unique(np.hstack([Ani,Bni]))
        mrg.drop(index=ABni,inplace=True)
        mrg.reset_index(inplace=True,drop=True)
        if len(mrg) == 0:
            empty = True
            print(snum,inout_flag,' empty')
            continue
        # count no-nan obs before cleaning
        nobs_b4.append(mrg.shape[0])
        nAnans.append(len(Ani[0]))
        nBnans.append(len(Bni[0]))

        # we will count paired obs at the end, but don't need the number repeated for the indoor and outdoor monitors
        # make npairs for 'in' be nans, we will add the count on the outdoor line
        if inout_flag == 'in':
            npairs.append(np.nan)

        # check that there is a B sensor (in previous code we made PMB variables -777.7 if there wasn't a B sensor)
        if mrg['PMB_atm'].max() == -777.7:
            # this monitor doesn't have a B sensor
            flag.append(-777.7)
            b4stats = [-777.7,-777.7,-777.7,-777.7,-777.7]
            MAE_b4.append(-777.7)
            indsrmv1 = []
            nPMrm.append(np.nan)
        else: # calc stats and filter by A and B allignment
            flag.append(333.3)
            # use cf=1 values (labels are switched in these files from Bonne)
            b4stats = linregress(mrg['PMA_atm'],mrg['PMB_atm'])
            # remove obs based on plantower manual reasonable agreement rules
            obs_diff = abs(mrg['PMA_atm'].values - mrg['PMB_atm'].values)
            pct_diff = 100.0*(obs_diff/mrg['PMA_atm'].values)
            MAE_b4.append(np.nanmean(obs_diff))
            # identify inds where PM <100, here A and B have to be within 10 ug/m3
            less100_inds = np.where(mrg['PMA_atm'].values<100.0)[0]
            indsrmv1a = less100_inds[np.where(obs_diff[less100_inds] > 10.0)[0]]
            # identify inds where PM > 100, here A and B have to be within 10% agreement
            gr100_inds = np.where(mrg['PMA_atm'].values>=100.0)[0]
            indsrmv1b = gr100_inds[np.where(pct_diff[gr100_inds]>10.0)]
            indsrmv1 = np.hstack([indsrmv1a,indsrmv1b])
            nPMrm.append(len(indsrmv1))

        # remove observations based on A-B agreement criteria
        mrg_clean1 = mrg.drop(indsrmv1,axis=0)
        mrg_clean1.reset_index(inplace=True,drop=True)
        
        # for A sensor, remove RH outside 0-99, T outside 14 to 140 F (from plantower manual)
        indsC1a = np.where(mrg_clean1['T_A'] > 140.0)
        indsC1b = np.where(mrg_clean1['T_A'] < 14.0)
        indsC1 = np.hstack([indsC1a,indsC1b])
        indsD1 = np.where(mrg_clean1['RH_A'] > 99.0)
        indsE1 = np.where(mrg_clean1['RH_A'] < 0.0)

        # log these for cleaning metadata
        nTArm.append(len(indsC1[0]))
        nRHArm.append(len(indsD1[0]) + len(indsE1[0]))

        # check B sensor for T and RH
        # don't use this ... the T is weird and removes a majority of data points
        nTBrm.append(np.nan)
        nRHBrm.append(np.nan)
        
        indsrmv2 = np.unique(np.hstack([indsC1,indsD1,indsE1]))
        mrg_clean2 = mrg_clean1.drop(indsrmv2,axis=0)
        mrg_clean2.reset_index(inplace=True,drop=True)

        # remove PM > 500.0
        inds_Armv = np.where(mrg_clean2['PMA_atm']>500.0)
        inds_Brmv = np.where(mrg_clean2['PMB_atm']>500.0)
        inds_rmv = np.unique(np.hstack([inds_Armv[0],inds_Brmv[0]]))
        nPMhigh_rm.append(len(inds_rmv))
        mrg_clean = mrg_clean2.drop(inds_rmv)
        mrg_clean.reset_index(inplace=True,drop=True)
        
        if len(mrg_clean)==0:
            after_stats = [np.nan]*5
            b4stats_all = np.vstack([b4stats_all,np.array(b4stats)])
            after_stats_all = np.vstack([after_stats_all,np.array(after_stats)])
            nobs_after.append(0)
            MAE_after.append(np.nan)
            start_time.append(np.nan)
            stop_time.append(np.nan)
            PM_means.append(np.nan)
            mis_clean.append(np.nan)
            empty = True
            print(mi,'clean empty')
            if inout_flag == 'out': # already made all in flags nans
                npairs.append(np.nan)
            continue

        # check filters worked
        if mrg_clean['PMA_atm'].max()> 500.0:
            print('error, check removal of high PM, A')
            hi=bye # exit code.... I know there are better ways to do this, but this works. if it ain't broke...
        if mrg_clean['PMB_atm'].max()>500.0:
            print('error, check removal of high PM, B')
            hi=bye
        if mrg_clean['T_A'].min()<14.0:
            print(mi,'T not filtered, min')
            hi=bye
        if mrg_clean['T_A'].max() > 140.0:
            print(mi,'T not filtered, max')
            hi=bye
        if mrg_clean['RH_A'].min() < 0.0:
            print(mi,'RH not filtered, min')
            hi=bye
        if mrg_clean['RH_A'].max() > 99.0:
            print(mi,'RH not filtered, max')
            hi=bye

        # calculate stats for monitors with a B sensor and add correction
        # check that there is a B sensor
        if mrg['PMB_cf1'].max()==-777.7:
            after_stats = [-777.7,-777.7,-777.7,-777.7,-777.7]
            MAE_after.append(-777.7)
            # add Barkjohn correction, remember atm and cf1 are flipped
            PMcf_1 = mrg_clean['PMA_atm'].values
            PMatm = mrg_clean['PMA_cf1'].values
            PM25_bj = 0.524*PMcf_1 - 0.0862*mrg_clean['RH_A'] + 5.75
            # add Delp and Singer correction for indoor, for testing different corrections
            PM25_ds = 0.48*PMcf_1
            # add LRAPA correction, for testing different corrections
            PM25_lrapa = 0.5*PMcf_1 - 0.66
            
        else: # calc stats and average A-B sensors for correcctions
            after_stats = linregress(mrg_clean['PMA_atm'],mrg_clean['PMB_atm'])
            obs_diff_after = abs(mrg_clean['PMA_atm'].values - mrg_clean['PMB_atm'].values)
            MAE_after.append(np.mean(obs_diff_after))
            # add Barkjohn correction
            PMcf_1 = np.mean([mrg_clean['PMA_atm'].values,mrg_clean['PMB_atm'].values],axis=0)
            PMatm = np.mean([mrg_clean['PMA_cf1'].values,mrg_clean['PMB_cf1'].values],axis=0)
            PM25_bj = 0.524*PMcf_1 - 0.0862*mrg_clean['RH_A'] + 5.75
            # add Delp and Singer correction for indoor, for testing different corrections
            PM25_ds = 0.48*PMcf_1
            # add LRAPA correction, for testing different corrections
            PM25_lrapa = 0.5*PMcf_1 - 0.66

        ################################################################################
        # save stats 
        ################################################################################
        b4stats_all = np.vstack([b4stats_all,np.array(b4stats)])
        after_stats_all = np.vstack([after_stats_all,np.array(after_stats)])
        nobs_after.append(mrg_clean.shape[0])

        ################################################################################
        # create clean mrg
        ################################################################################
        # save clean merge with corrected names
        mrg_clean['PMcf_1_avg'] = PMcf_1
        mrg_clean['PMatm'] = PMatm
        mrg_clean['PM25_bj'] = PM25_bj
        mrg_clean['PM25_ds'] = PM25_ds
        mrg_clean['PM25_LRAPA'] = PM25_lrapa
        
        # calc PM mean to add to metadata
        PM_mean = np.nanmean(mrg_clean['PM25_bj'].values)
        PM_means.append(PM_mean)
        ################################################################################
        # save time bounds to meta data
        ################################################################################    
        start_time.append(mrg_clean['time'].iloc[0])
        stop_time.append(mrg_clean['time'].iloc[-1])
        mis_clean.append(snum+'_'+inout_flag)
        
        ################################################################################
        # write to file
        ################################################################################
        # in loop: write mrg to file
        mrg_clean.to_csv(out_fp + 'purpleair_PMAB20_mID'+snum+'_AnBmrg_wUScoloc_d1000m_new_bfix_'+file_desc+'_'+inout_flag+'_2020.csv')
        print(snum,inout_flag,'saved')
        if inout_flag=='in':
            in_mrg_clean = pd.DataFrame(data = {'time':pd.to_datetime(mrg_clean['time'].values)})
            in_mrg_clean['PM25_bj_in'] = mrg_clean['PM25_bj']
            in_mrg_clean['PM25_ds_in'] = mrg_clean['PM25_ds']
            in_mrg_clean['PM25_LRAPA_in'] = mrg_clean['PM25_LRAPA']
            in_mrg_clean['PMcf_1_avg_in'] = PMcf_1
            in_mrg_clean['PMatm_in'] = PMatm
            in_mrg_clean['T_in'] = mrg_clean['T_A']
            in_mrg_clean['RH_in'] = mrg_clean['RH_A']
        elif inout_flag == 'out':
            out_mrg_clean = pd.DataFrame(data={'time':pd.to_datetime(mrg_clean['time'].values)})
            out_mrg_clean['PM25_bj_out'] = mrg_clean['PM25_bj']
            out_mrg_clean['PM25_ds_out'] = mrg_clean['PM25_ds']
            out_mrg_clean['PM25_LRAPA_out'] = mrg_clean['PM25_LRAPA']
            out_mrg_clean['PMcf_1_avg_out'] = PMcf_1
            out_mrg_clean['PMatm_out'] = PMatm
            out_mrg_clean['T_out'] = mrg_clean['T_A']
            out_mrg_clean['RH_out'] = mrg_clean['RH_A']
    # merge in and out data
    if not empty:
        inoutmrg = pd.merge_asof(in_mrg_clean,out_mrg_clean,on='time',tolerance = dt.timedelta(minutes=1),direction='nearest')
        inoutmrg.to_csv(out_fp_inout +  'purpleair_PMAB20_mID'+snum+'_AnBmrg_wUScoloc_d1000m_new_bfix_2020_'+file_desc+'.csv')
        # count paired obs
        out_nans = np.where(np.isnan(inoutmrg['PM25_bj_out']))[0]
        in_nans = np.where(np.isnan(inoutmrg['PM25_bj_in']))[0]
        rmv_pairs = np.unique(np.hstack([out_nans,in_nans]))
        npairs.append(inoutmrg.shape[0] - len(rmv_pairs))
        print('total paired obs:',inoutmrg.shape[0]-len(rmv_pairs))
        '''
        # plot to check
        fig,axarr = plt.subplots(1,2)#,subplot_titles=['inside','outside'])
        axarr[0].scatter(in_mrg_clean['time'],in_mrg_clean['PM25_bj_in'],s=1)
        axarr[0].scatter(inoutmrg['time'],inoutmrg['PM25_bj_in']+200,s=1)
        axarr[1].scatter(out_mrg_clean['time'],out_mrg_clean['PM25_bj_out'],s=1)
        axarr[1].scatter(inoutmrg['time'],inoutmrg['PM25_bj_out']+50,s=1)
        fig.show()
        '''
# out of loop: write updated metadata to file
clean_stats = pd.DataFrame(data={'ID':mis_clean,'start_time':start_time,'stop_time':stop_time,'PM_mean':PM_means,'nPMrm':nPMrm,'nPMhigh_rm':nPMhigh_rm,'nTArm':nTArm,'nRHArm':nRHArm,'nTBrm':nTBrm,'nRHBrm':nRHBrm,'nAnans':nAnans,'nBnans':nBnans,'site_rval_b4':b4stats_all[1:,2],'site_slope_b4':b4stats_all[1:,0],'site_MAE_b4':MAE_b4,'site_rval_after':after_stats_all[1:,2],'site_slope_after':after_stats_all[1:,0],'site_MAE_after':MAE_after,'nobs_b4':nobs_b4,'nobs_after':nobs_after,'type_flag':flag,'npairs':npairs})

clean_stats.to_csv(out_fp + 'clean_stats_test_2020_nTRH_wUS_'+file_desc+'.csv')
print('cleaning done. metadata saved.')
################################################################################
# make figures
################################################################################
'''
# A/B cleaning
fig, axarr = plt.subplots(1,2)
axarr.flatten()
axarr[0].plot(mrg['PMA_atm'],mrg['PMB_atm'],'.')
axarr[0].plot(np.arange(0,mrg['PMA_atm'].max()),np.arange(0,mrg['PMA_atm'].max()))
axarr[0].set_title(str(mi)+inout_flag+'\n 1-to-1 before clean')
axarr[0].set_xlabel('PMA_atm')
axarr[0].set_ylabel('PMB_atm')

axarr[1].plot(mrg_clean['PMA_atm'],mrg_clean['PMB_atm'],'.')
axarr[1].plot(np.arange(0,mrg_clean['PMA_atm'].max()),np.arange(0,mrg_clean['PMA_atm'].max()))
axarr[1].set_title(str(mi)+inout_flag+'\n 1-to-1 after clean')
axarr[1].set_xlabel('PMA_atm')
axarr[1].set_ylabel('PMB_atm')
fig.show()

# timeseries of og cf1 and corrected
fig,ax = plt.subplots(1)
ax.plot(mrg_clean['time'],PM25_bj,label='Barkjohn CF')
ax.plot(mrg_clean['time'],PMcf_1,label='CF1, AB mean')
ax.set_title(str(mi)+inout_flag+' \n timeseries')
ax.set_xlabel('time')
ax.set_ylabel('PM2.5 [ug/m3]')
ax.legend()
fig.show()

'''
