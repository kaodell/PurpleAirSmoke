#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_check_sesnor_lists.py
    python script to cross-compare two sensor lists we have from the PurpleAir dataset
    downloaded at different times to check for monitor location consistency
Created on Tue Apr  6 08:06:45 2021
@author: kodell
"""
#%% user inputs
# path to project folder containing the sensor lists and where to place output
proj_folder = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/'
bonne_sl_fn = proj_folder + 'PA_data/purpleair_sitelist_global_4.csv'
jude_sl_fn = proj_folder + 'PA_data/Jude_sensor_metadata.csv'
out_fp = proj_folder + 'PA_data/'
out_desc = 'fv' # final version
#%% user-defined functions
# haversine function from Will Lassman
def haversine(lon0,lon1,lat0,lat1):
    r = 6371000.#m                                                                                                                                                                                                                                                 
    lon0 = lon0*np.pi/180
    lon1 = lon1*np.pi/180
    lat0 = lat0*np.pi/180
    lat1 = lat1*np.pi/180
    
    return 2*r*np.arcsin(np.sqrt(np.sin((lat1 - lat0)/2.)**2 +\
		 np.cos(lat0)*np.cos(lat1)*np.sin((lon1 - lon0)/2.)**2))

#%% load modules - versions listed in the local python environment file
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "chrome"
pio.templates.default = "seaborn"
mapbox_access_token='pk.eyJ1Ijoia2FvZGVsbCIsImEiOiJjanZza3k1bGkzMHZoNDhwYjdybTYyOTliIn0.KJyzHWVzu2U087Ps215_LA'

#%% load data
bonne_df = pd.read_csv(bonne_sl_fn,dtype={'thingspeak_primary_id':np.int64})
jude_df = pd.read_csv(jude_sl_fn,dtype={'thingspeak_primary_id':np.int64})

#%% plot monitors in both lists
fig = px.scatter_mapbox(bonne_df, lat="lat", lon="lon", color="thingspeak_primary_id",zoom=10,
                        hover_name="lastseen")
fig.update_layout(title = 'monitors in Bonne list',
                  mapbox=dict(accesstoken=mapbox_access_token,bearing=0,
        center=dict(lat=38.92,lon=-77.07),pitch=0,zoom=4),)
fig.show()

fig = px.scatter_mapbox(jude_df, lat="lat", lon="lon", color="thingspeak_primary_id",zoom=10,
                        hover_name="lastseen")
fig.update_layout(title = 'monitors in Jude list',
                  mapbox=dict(accesstoken=mapbox_access_token,bearing=0,
        center=dict(lat=38.92,lon=-77.07),pitch=0,zoom=4),)
fig.show()

#%% match IDs and compare lat/lons in the two sensor lists
count_move = 0 # there should be 535 moved monitors, fv, 10 Jan 2022
dist_move = []
ID_move = []
count_loc_change = 0 # there should be 66 monitors with a location change, fv, 10 Jan 2022

for i in range(jude_df.shape[0]):
    # get ID from Jude's list and ind for matching ID in Bonne's list
    itID = jude_df['thingspeak_primary_id'].iloc[i]
    bind = np.where(bonne_df['thingspeak_primary_id']==itID)[0]
    # some IDs are only in Bonne's list, we'll get a list of these in the next section 
    if len(bind)==0:
        continue
    # check if monitor has moved
    bind = bind[0]
    dist = haversine(jude_df['lon'].iloc[i], bonne_df['lon'].iloc[bind],
            jude_df['lat'].iloc[i], bonne_df['lat'].iloc[bind])
    if dist > 100:
        count_move += 1
        dist_move.append(dist)
        ID_move.append(itID)
    
    # check if location has changed - only if this is a 'A' sesor
    # B sensors don't have locations in Jude's file
    if np.isnan(jude_df['parentid'].iloc[i]): # paretid listed as NaN for A sensors
        if jude_df['device_locationtype'].iloc[i]=='outside':
            if bonne_df['device_locationtype'].iloc[bind] == 1:
                ID_move.append(itID)
                dist_move.append(-10) # -10 to indicate this is change in location flag not lat/lon
                count_loc_change += 1
            elif bonne_df['device_locationtype'].iloc[bind] != 0:
                print('bonne loc type error at ',bind)
        elif jude_df['device_locationtype'].iloc[i]=='inside':
            if bonne_df['device_locationtype'].iloc[bind] == 0:
                ID_move.append(itID)
                dist_move.append(-10)
                count_loc_change += 1
            elif bonne_df['device_locationtype'].iloc[bind] != 1:
                print('bonne loc type error at ',bind)
    else:
        # this is a B sensor, do nothing
        continue

#%% plot moved monitors
lon = []
lat = []
for ID in ID_move:
    ind = np.where(jude_df['thingspeak_primary_id'].values==ID)[0][0]
    lon.append(jude_df['lon'].iloc[ind])
    lat.append(jude_df['lat'].iloc[ind])
df = pd.DataFrame(data={'lon':lon,'lat':lat,'dist_move':dist_move})

fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="dist_move",
                  color_continuous_scale=px.colors.cyclical.IceFire,zoom=10)
fig.update_layout(autosize=True,hovermode='closest',
    mapbox=dict(accesstoken=mapbox_access_token,bearing=0,center=dict(lat=38.92,lon=-77.07),
                pitch=0,zoom=4),)
fig.show()

#%% now loop through and find any IDs in Jude's list but not Bonne's (and vice versa)
missing_IDs_Bonne = [] # IDs missing from Bonne's list that are in Jude's
for i in range(jude_df.shape[0]):
    ID = jude_df['thingspeak_primary_id'].iloc[i]
    bind = np.where(bonne_df['thingspeak_primary_id'].values==ID)[0]
    if len(bind)==0:
        missing_IDs_Bonne.append(ID)
missing_IDs_Jude = [] # IDs missing from Jude's list that are in Bonne's
for i in range(bonne_df.shape[0]):
    ID = bonne_df['thingspeak_primary_id'].iloc[i]
    jind = np.where(jude_df['thingspeak_primary_id'].values==ID)[0]
    if len(jind)==0:
        missing_IDs_Jude.append(ID)
print(len(missing_IDs_Bonne),' IDs in Jude list but not Bonne') # 949 as of fv, 01.10.22
print(len(missing_IDs_Jude),' IDs in Bonne list but not Jude') # 29481 as of fv, 01.10.22

#%% investigate IDs missing from Bonne's list
# we ultimately use the data from Bonne (most recent download), so we don't have
# up-to-date data for these sensors in Jude's list but not hers
# let's figure out what's up with these monitors missing from Bonne's list

lon = []
lat = []
names = []
parent_id = []
flags = []
invalidB_inds = []

for i in range(len(missing_IDs_Bonne)):
    ID = missing_IDs_Bonne[i]
    ind = np.where(jude_df['thingspeak_primary_id'].values==ID)[0][0]
    lon.append(jude_df['lon'].iloc[ind])
    lat.append(jude_df['lat'].iloc[ind])
    name = jude_df['label'].iloc[ind]
    names.append(name)
    flag = np.isnan(jude_df['parentid'].iloc[ind]) # see if labeled as a 'B' sensor
    flags.append(flag)

    # check if there should be a B sensor at all
    if ~flag: # flag is FALSE for B sensors
        # get parent ID (A sensor) - device details not copied for the 'B' sensor
        jid_a = jude_df['parentid'].iloc[ind]
        parent_id.append(jid_a)
        aind = np.where(jude_df['id'].values==jid_a)[0][0]        
        if jude_df['type'].iloc[aind][:7] == 'PMS1003':
            # this monitor type doesn't have a B sensor
            invalidB_inds.append(i)
        else:
            # see if the aid is in Bonne's file and how it is labelled
            aind_Bonne = np.where(bonne_df['id']==jid_a)[0]
            if len(aind_Bonne)==1:
                if bonne_df['type'].iloc[aind_Bonne[0]]=='PA-I':
                    # doesn't have B sensor
                    invalidB_inds.append(i)
                elif bonne_df['type'].iloc[aind_Bonne[0]]=='UNKNOWN':
                    # we can't include these in our analysis anyway
                    invalidB_inds.append(i)
    else:
        parent_id.append(jude_df['id'].iloc[ind])
        
invalidB_inds = np.array(invalidB_inds)
df = pd.DataFrame(data={'lon':lon,'lat':lat,'id':missing_IDs_Bonne,'name':names,'flag':flags,
                        'parentid':parent_id})
df_actual = df.drop(invalidB_inds)
df_actual.reset_index(inplace=True,drop=True)

print('n sensors missing',len(missing_IDs_Bonne)-len(invalidB_inds),
      'n moniotrs missing',len(df_actual['parentid'].unique()))

#%% plot missing monitors monitors
fig = px.scatter_mapbox(df_actual, lat="lat", lon="lon",zoom=10,
                        hover_name="id")
fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(accesstoken=mapbox_access_token,bearing=0,
        center=dict(lat=38.92,lon=-77.07),pitch=0,zoom=4),)
fig.show()

# I investigated these missing monitors by hand, see project notebook
# notes on 21 November 2021

#%% save data
# write IDs that have moved to a file and we can remove from the metadata file
ID_move_df = pd.DataFrame(data={'ID':ID_move,'distance_moved':dist_move})
ID_move_df.to_csv(out_fp+'IDs_moved_'+out_desc+'.csv')








