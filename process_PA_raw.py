'''
process_PA_raw.py
python3 script to read purple air raw files, match A&B sensors, and save to process 
in another code
v0 - 01.19.21 - script created
v1 - 02.22.21 - edited to make merge in this code 
v2 - 03.10.21 - edited to work with all PA sensors
v2.1 04.01.21 - fixing to not require in B sensor and flag those that don't (these are inside mons)
v3 - 04.16.21 - edited to work with PA data downloaded by Bonne and upated sensor list with monitor types
Katelyn O'Dell
'''
################################################################################
# User Inputs
################################################################################
# specify file path
PA_fp = '/home/bford/purpleair/k_select/'

# metadata file name (created on local machine, has co-loc monitor IDs and corresponding A/B IDs)
# fv = final version
PAmd_fn = '/fischer-scratch/kaodell/purple_air/sensor_lists/wUS_co_loc_sensor_list_1000m_Bonne_global4_fv.csv'

# out file and fig paths
out_file_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/ABmrg/'
out_desc = '_wUScoloc_d1000m_fv'
################################################################################
# Load Modules
################################################################################
import pandas as pd
import numpy as np
from os import listdir
import os
import matplotlib.pyplot as pl
import plotly.graph_objects as go
import scipy.stats as st
import datetime
################################################################################
# user defined functions
################################################################################
def mkPAdf(filename, test_names, inout_flag, AB_flag):
    T = True
    RH = True
    # load file
    pa_df = pd.read_csv(filename,dtype={'created_at':'string','pm1_0_atm':float,
                                        'pm2_5_atm':float,'pm10_0_atm':float,
                                        'pm2_5_cf_1':float,'temperature':float,
                                        'humidity':float})
    # check if file empty
    if pa_df.shape[0] == 0:
        return np.nan, np.nan, 'empty file'
    
    # check variable names
    hname = test_names[0]
    tname = test_names[1]
    cf1_name = test_names[2]
    atm_name = test_names[3]
    # PM_cf1 - looks like all the files from Bonne have the same cf1 name
    if cf1_name not in pa_df.keys():
        print(pa_df.keys())
        return np.nan, np.nan,'check cf1 name'
    # PM_atm - same as above
    if atm_name not in pa_df.keys():
        print(pa_df.keys())
        return np.nan,np.nan, 'check atm name'
    # relative humidity, may not be in some B files
    if hname not in pa_df.keys():
        RH = False
    # temperature, may not be in some B files
    if tname not in pa_df.keys():
        T = False
    # create datetime
    pa_dt = []
    count = 0
    for t in pa_df['created_at']:
        t = str(t)
        if len(t)<20:
            count += 1
            pa_dt.append(np.nan)
            print('time string too short for',count,'observation(s)')
            continue
        dt = datetime.datetime.strptime(t[:-1],'%Y-%m-%dT%H:%M:%S')
        pa_dt.append(dt)
    pa_dt = np.array(pa_dt)
        
    # now treat A and B sensors seprately
    # A file should always have T and RH
    if AB_flag == 'A':
        if not(RH):
            return np.nan, np.nan,'check A humidity name'
        elif not(T):
            return np.nan, np.nan,'check A temp name'
        PAmrg1 = pd.DataFrame(data={'time':pa_dt,'PMA_atm':pa_df[atm_name].values,'PMA_cf1':pa_df[cf1_name].values,\
                                    'RH_A':pa_df[hname].values,'T_A':pa_df[tname].values,'atime':pa_dt})
    
    # B file can not have T or RH - we ultimately use the values from the A sensor file because these are sometimes weird
    if AB_flag == 'B':
        if all([RH,T]): # have T and RH
            PAmrg1 = pd.DataFrame(data={'time':pa_dt,'PMB_atm':pa_df[atm_name].values,'PMB_cf1':pa_df[cf1_name].values,\
                                    'RH_B':pa_df[hname].values,'T_B':pa_df[tname].values,'btime':pa_dt})
        elif any([RH,T]):
            if T: # have T but not RH
                PAmrg1 = pd.DataFrame(data={'time':pa_dt,'PMB_atm':pa_df[atm_name].values,'PMB_cf1':pa_df[cf1_name].values,\
                                            'RH_B':[-999.9]*pa_df.shape[0],'T_B':pa_df[tname].values,'btime':pa_dt})
            else: # have RH but not T
                PAmrg1 = pd.DataFrame(data={'time':pa_dt,'PMB_atm':pa_df[atm_name].values,'PMB_cf1':pa_df[cf1_name].values,\
                                    'RH_B':pa_df[hname].values,'T_B':[-999.9]*pa_df.shape[0],'btime':pa_dt})
        else: # have neither T nor RH
            PAmrg1 = pd.DataFrame(data={'time':pa_dt,'PMB_atm':pa_df[atm_name].values,'PMB_cf1':pa_df[cf1_name].values,\
                                    'RH_B':[-999.9]*pa_df.shape[0],'T_B':[-999.9]*pa_df.shape[0],'btime':pa_dt})

    # drop duplicates and reset index
    PAmrg = PAmrg1.drop_duplicates(keep='last',ignore_index=True)
    PAmrg_t = PAmrg1.drop_duplicates(subset='time',keep='last')
    if PAmrg1.shape[0] - PAmrg.shape[0] > 0:
        print(PAmrg1.shape[0]-PAmrg.shape[0],'full duplicates dropped')
    if PAmrg.shape[0] != PAmrg_t.shape[0]:
        print(PAmrg.shape[0]-PAmrg_t.shape[0],'time duplicates do not match')
        # keeping time duplicates and will average in the averages... dont know which timestamp/observation is correct
    # ensuring points with the same time are in fact duplicate obs .. seems to happen a couple times in a lot of files where there is a duplicate time but the obs don't match. just keep and use in the averages.
    PAmrg.reset_index(inplace=True,drop=True)

    # remove locations where time is nan
    tnan_inds = np.where(pd.isna(PAmrg['time']))[0]
    PAmrg_tfix = PAmrg.drop(tnan_inds)
    if len(tnan_inds) >0:
        print(len(tnan_inds),'time is nan. dropping')
    PAmrg_tfix.reset_index(inplace=True,drop=True)
    
    # add datetime to loaded array for plotting
    pa_df_out=pa_df.drop(tnan_inds)
    pa_df_out.reset_index(inplace=True,drop=True)
    pa_df_out['datetime'] = pd.to_datetime(pa_df_out['created_at'])
    
    # return dataframe
    return PAmrg_tfix, pa_df_out,'completed'
################################################################################
# Load metadata file and get list of sensors
################################################################################
md_df = pd.read_csv(PAmd_fn,dtype={'in_name':'string','in_AID':int,
                                   'in_AID_tp':int,'in_lat':float,'in_lon':float,
                                   'out_name':'string','out_AID':int,'out_AID_tp':int,
                                   'out_lat':float,'out_lon':float,'distance':float,'in_AID_tp_key':'string','out_AID_tp_key':'string','in_mon_type':'string',
                                   'out_mon_type':'string','in_BID':int,
                                   'out_BID_tp':int,'in_BID_tp':int,'in_BID_tp_key':'string','out_BID_tp_key':'string','county_names':'string','FIPS_code':float,'area_abbr':'string'})
flist = listdir(PA_fp)
################################################################################
# Load sensors in metadata file and create datetime arrays rounded to user input rtime
################################################################################
md_rmv_inds = []
processed_names = []
missing_file_names = []
missing_file_types = []
all_nan_fn = []
all_nan_ftype = []
ftype = np.array(['inA','inB','outA','outB'])
for i in range(md_df.shape[0]):
#for i in range(300,301): # for testing
    # create sensor file name for each co-loc A and B sensor
    inAstr = 'sensor-'+str(md_df['in_AID_tp'].iloc[i])+'.csv'
    inBstr = 'sensor-'+str(md_df['in_BID_tp'].iloc[i])+'.csv' 
    outAstr = 'sensor-'+str(md_df['out_AID_tp'].iloc[i])+'.csv'
    outBstr = 'sensor-'+str(md_df['out_BID_tp'].iloc[i])+'.csv'

    # check monitor types - whether or not we need a B sensor and to remove inside monitors that are outside
    # first chenck indoor monitor type
    if md_df['in_mon_type'].iloc[i] in ['PA-II','PA-II-Flex','PA-II-SD']: # outside monitor is inside, must have a B sensor
        snames_check = np.array([inAstr, outAstr, inBstr, outBstr])
        ftype = np.array(['inA','outA','inB','outB'])

    elif md_df['in_mon_type'].iloc[i] == 'PA-I': # inside monitor is inside, don't need a B sensor
        snames_check = np.array([inAstr,outAstr,outBstr])
        ftype = np.array(['inA','outA','outB'])
    else:
        processed_names.append('NA')
        if md_df['in_mon_type'].iloc[i]!= 'UNKNOWN':
            print('error - inside monitor type not recognized:',md_df['in_mon_type'].iloc[i])
        continue

    # next, check outdoor monitor type
    if md_df['out_mon_type'].iloc[i] == 'PA-I':
        processed_names.append('NA')
        # these should already be flagged
        if md_df['out_BID'].iloc[i] + 777.0 != 0:
            print('!!!!!!!!SENSOR LIST ERROR - CHECK!!!!!!!!!')
        print('in monitor outside, removing')
        continue
        
    # look for each of the file names we need in the file list
    inflist_bool = []
    for sname in snames_check:
        inflist_bool.append(sname in flist)

    # if any are missing, save ind and move on
    if not all(inflist_bool):
        md_rmv_inds.append(i)
        minds = np.where(~np.array(inflist_bool))[0]
        missing_file_names=np.hstack([missing_file_names,snames_check[minds]])
        missing_file_types=np.hstack([missing_file_types,ftype[minds]])
        processed_names.append('NA')
        print(i,'missing files')
        continue
    
    # now that we have identified the files we need and checked for missing files,
    # load data and place in dataframe
    var_names = ['humidity','temperature','pm2_5_cf_1','pm2_5_atm']
    inAmrg, inAload, inAmsg = mkPAdf(PA_fp+inAstr, var_names, 'in', 'A')
    outAmrg, outAload,outAmsg = mkPAdf(PA_fp+outAstr, var_names, 'out', 'A')
    outBmrg, outBload,outBmsg = mkPAdf(PA_fp+outBstr, var_names, 'out', 'B')

    # load B file inside if this an outside monitor and check that we got all sensors processed
    if md_df['in_mon_type'].iloc[i] != 'PA-I': # need a B sensor
        inBmrg, inBload,inBmsg = mkPAdf(PA_fp+inBstr, var_names, 'in', 'B')
        if all([inAmsg=='completed',outAmsg=='completed',inBmsg=='completed',outBmsg=='completed']):
            process = True
        else:
            process = False
    else:
        inBmsg = 'not needed'
        if all([inAmsg=='completed',outAmsg=='completed',outBmsg=='completed']):
            process = True
        else:
            process = False

    print('inA:',inAmsg,'inB:',inBmsg,'outA:',outAmsg,'outB:',outBmsg)
    
    # if we got all the sensors we needed, create A-B merges for the indoor and outdoor monitors and save
    if process:
        # merge A and B sensors outside
        out_site_mrg = pd.merge_asof(outAmrg,outBmrg,on='time',tolerance = datetime.timedelta(minutes=1),direction='nearest')
        # check for inside B sensor
        if inBmsg == 'completed':
            in_site_mrg = pd.merge_asof(inAmrg,inBmrg,on='time',tolerance = datetime.timedelta(minutes=1),direction='nearest')
        if inBmsg == 'not needed':
            in_site_mrg = inAmrg
            in_site_mrg['PMB_atm'] = -777.7
            in_site_mrg['PMB_cf1'] = -777.7

        # save files
        in_site_mrg.to_csv(out_file_fp+'purpleair_colocmID'+str(i).zfill(4)+'_mrg'+out_desc+'_in.csv')
        out_site_mrg.to_csv(out_file_fp+'purpleair_colocmID'+str(i).zfill(4)+'_mrg'+out_desc+'_out.csv')
        processed_names.append(i)
        print('ind',i,'in and out saved')
        
        '''
        # plot to check
        fig,axarr = pl.subplots(1,2)#,subplot_titles=['inside A','inside B'])
        axarr[0].plot(inAload['datetime'],inAload['pm2_5_atm'],'.')
        axarr[0].plot(in_site_mrg['atime'],in_site_mrg['PMA_atm']+200,'.')
        axarr[1].plot(outBload['datetime'],outBload['pm2_5_atm'],'.')
        axarr[1].plot(out_site_mrg['btime'],out_site_mrg['PMB_atm']+200,'.')
        fig.show()

        # plot to check
        fig,axarr = pl.subplots(1,2)#,subplot_titles=['inside A','inside B'])
        axarr[0].plot(outAload['datetime'].iloc[:5000],outAload['pm2_5_atm'].iloc[:5000],'.')
        '''
    # if all the files didn't process, save ind and continue
    else:
        md_rmv_inds.append(i)
        processed_names.append('NA')
        print('ind',i,'error')
        
# save meta data
print(len(np.unique(md_rmv_inds)),'site pairs unable to be processed')
md_df['processed_names'] = processed_names
md_df.to_csv(out_file_fp+'PAmetadata_wprocessed_names'+out_desc+'.csv')
print('new meta data saved')

# save info on missing files
md_rmv_inds_df = pd.DataFrame(data={'coloc_rmv_inds':md_rmv_inds})
md_rmv_inds_df.to_csv(out_file_fp+'coloc_missing_inds'+out_desc+'.csv')

missing_files_df = pd.DataFrame(data={'missing_file_name':missing_file_names,
                                      'missing_file_type':missing_file_types})
missing_files_df.to_csv(out_file_fp+'missing_file_names'+out_desc+'.csv')
