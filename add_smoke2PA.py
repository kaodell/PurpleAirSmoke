'''
add_smoke2PA.py
python script to load purple air files and add smoke binary column
v0 - initial code, 02.24.21
written by Katelyn O'Dell
'''
##################################################################
# user inputs
##################################################################
file_desc = 'fv' # fv = final version
scratch_fp = '/fischer-scratch/kaodell/purple_air/Bonne_data/final_files/' # folder on the scratch directory with the files
PA_fp = scratch_fp + 'inout_mrg_clean/' # purple air files to load - cleaned, merged indoor-outdoor data from previous code
HMS_fp = '/pierce-scratch/bford/satellite_data/HMS/smoke/' # file path to HMS smoke plumes
metadata_fn = scratch_fp + 'ABmrg/PAmetadata_wprocessed_names_wUScoloc_d1000m_'+file_desc+'.csv' # file metadata
out_fp = scratch_fp +'ABmrg_clean_wsmoke/'
year_use = 2020 

##################################################################
# import modules
##################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pytz
import shapefile
import matplotlib as mplt
import os
from timezonefinder import TimezoneFinder

##################################################################
# load metadata
##################################################################
all_md_df = pd.read_csv(metadata_fn,dtype={'in_name':'string','in_AID':int,
                                   'in_AID_tp':int,'in_lat':float,'in_lon':float,
                                   'out_name':'string','out_AID':int,'out_AID_tp':int,
                                   'out_lat':float,'out_lon':float,'distance':float,'in_AID_tp_key':'string','out_AID_tp_key':'string','in_mon_type':'string',
                                   'out_mon_type':'string','in_BID':int,
                                   'out_BID_tp':int,'in_BID_tp':int,'in_BID_tp_key':'string','out_BID_tp_key':'string','county_names':'string','FIPS_code':float,'area_abbr':'string','processed_names':float})

##################################################################
# loop through and load files and add HMS
##################################################################
for f_ID in all_md_df['processed_names']:
    if pd.isna(f_ID):
        continue
    # look for filename in the inout merge folder
    fn = 'purpleair_PMAB20_mID'+str(int(f_ID)).zfill(4)+'_AnBmrg_wUScoloc_d1000m_new_bfix_2020_'+file_desc+'.csv'
    if fn not in os.listdir(PA_fp):
        # no file available (data removed in cleaning or in-out pairing)
        continue

    # get lat/lon from metadata
    mind = np.where(all_md_df['processed_names']==f_ID)[0][0]
    slon = all_md_df['in_lon'].iloc[mind]
    slat = all_md_df['in_lat'].iloc[mind]

    # load merged in-out file
    sdf = pd.read_csv(PA_fp +fn,dtype={'time':'string','PM25_bj_in':float,'PM25_ds_in':float,'PM25_LRAPA_in':float,'PMcf_1_avg_in':float,'PMatm_in':float,'T_in':float,'RH_in':float,'PM25_bj_out':float,'PM25_ds_out':float,'PM25_LRAPA_out':float,'PMcf1_avg_out':float,'PMatm_out':float,'T_out':float,'RH_out':float})

    # convert datetime to local time
    # identify local timezone
    tzname = TimezoneFinder().timezone_at(lng=slon,lat=slat)
    sdf['datetime_UTC'] = pd.to_datetime(sdf['time'])
    datetime_locs = []
    for datetime in sdf['datetime_UTC']:
        datetime_utc = datetime.replace(tzinfo=pytz.utc)
        datetime_loc = datetime_utc.astimezone(pytz.timezone(tzname))
        datetime_locs.append(datetime_loc)
    sdf['datetime_loc'] = np.array(datetime_locs)
    
    # define start and stop times and days
    start_time_loc_t = sdf['datetime_loc'].iloc[0]
    stop_time_loc_t = sdf['datetime_loc'].iloc[-1]
    
    start_time_loc_day = dt.datetime(year = start_time_loc_t.year,
                                     month = start_time_loc_t.month,
                                     day = start_time_loc_t.day,
                                     hour=0,minute=0,second=0)
    stop_time_loc_day = dt.datetime(year= stop_time_loc_t.year,
                                    month=stop_time_loc_t.month,
                                    day=stop_time_loc_t.day,
                                    hour=23,minute=59,second=59)+dt.timedelta(days=1)

    # loop through days and load HMS
    # pre-allocate arrays
    sHMS = np.array([-999]*sdf.shape[0]) # HMS flag
    sevent_l = np.array([-999]*sdf.shape[0]) # smoke event length - ultimately don't use this in analysis
    # start smoke event len at zero
    yd_smoke = 0.0
    time = start_time_loc_day
    while time <= (stop_time_loc_day):
        # find dayinds in the file
        day_start = pytz.timezone(tzname).localize(time)
        day_end = pytz.timezone(tzname).localize(time + dt.timedelta(days=1))
        di = np.where((sdf['datetime_loc']>=day_start) & (sdf['datetime_loc']<day_end))
        di = np.array(di[0])

        # if day not in the file, reset event length to zero and move on to next day
        if len(di)==0:
            yd_smoke = 0.0
            time += dt.timedelta(days=1)
            continue

        # load HMS
        date_str = str(day_start.month).zfill(2) + str(day_start.day).zfill(2)
        # check HMS file is there
        HMS_fn = 'hms_smoke'+str(time.year)+date_str+'.dbf'
        if HMS_fn in os.listdir(HMS_fp + str(time.year)):
            # set zero to start, update to 1 if there is smoke
            sHMS[di] = 0
            # load file
            HMS_file = shapefile.Reader(HMS_fp + str(time.year) +'/'+ HMS_fn[:-4])
            plume_records = HMS_file.records()
            plume_shapes = HMS_file.shapes()
            # loop through shapefiles and see if they contain the site
            for j in range(len(plume_records)):
                plume_shp = plume_shapes[j]
                for i in range(len(plume_shp.parts)):
                    i0 = plume_shp.parts[i]
                    if i < len(plume_shp.parts)-1:
                        i1 = plume_shp.parts[i+1] - 1
                    else:
                        i1 = len(plume_shp.points)
                    seg = plume_shp.points[i0:i1+1]
                    mpath = mplt.path.Path(seg)
                    points = np.array([[slon,slat]])
                    mask = mpath.contains_points(points)[0]
                    if mask:
                        sHMS[di]=1
            # once we've looped through all the shapes, set event length
            sevent_l[di] = (yd_smoke + sHMS[di])*sHMS[di] # multipying by HMS ensures if its not a smoke day, event length resets to zero
            # reset yd smoke 
            yd_smoke = np.mean(sevent_l[di])
            
        else: # no HMS file this day, flag with -555
            sHMS[di]=-555
            sevent_l[di]=-555
            yd_smoke = 0.0
        time += dt.timedelta(days=1)
    print(np.unique(sevent_l))
    
    # save out to file
    sdf['HMS_flag']=sHMS
    sdf['event_len']=sevent_l
    sdf.to_csv(out_fp +'purpleair_PMAB20_mID'+str(int(f_ID)).zfill(4)+'_AnBmrg_wUScoloc_d1000m_clean_wsmoke_'+str(year_use)+'_'+file_desc+'.csv')
    print(f_ID,' saved')

    '''
    # plot to check, then compare this time series with online HMS files
    fig,ax = plt.subplots(1,1)
    ax.scatter(sdf['time'],sHMS)
    ax.scatter(sdf['time'],sevent_l)
    plt.show()
    '''
