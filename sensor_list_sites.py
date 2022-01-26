#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purple_Air_Sites.py
    python script to plot the purple air indoor and outdoor monitor locaitons
    in the sensor lists file, find co-located monitors, assign counties to monitors, and save inds
Created on Tue Mar  9 10:46:58 2021
@author: kodell
"""
#%% user inputs
proj_folder = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/'
# Bonne updated sensor list
sensor_list_fn = proj_folder + 'PA_data/purpleair_sitelist_global_4.csv'
# IDs that have moved between Jude and Bonne sensor lists, output from cross_check_sensor_lists.py
moved_IDs_fn = proj_folder + 'PA_data/IDs_moved_fv.csv'
# county shapefile for assigning county flags
#shapefile_fn_county = proj_folder + 'PA_data/cb_2018_us_county_500k/cb_2018_us_county_500k'
shapefile_fn_county = '/Volumes/ODell_Files/School_Files/CSU/Research/NSF/purple_air_smoke/PA_data/SVI2018_US_COUNTY/SVI2018_US_COUNTY'
# out fig fp
out_fig_path = proj_folder + 'PA_figures/'
# out data path
out_fp = proj_folder + 'PA_data/'
# max distance for co-located monitors in meters
max_dist = 1000.0 #m
# out file description 
out_desc = 'Bonne_global4_fv' # fv = 'final version' before submission

#%% import modules
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "chrome"
pio.templates.default = "seaborn"
mapbox_access_token='pk.eyJ1Ijoia2FvZGVsbCIsImEiOiJjanZza3k1bGkzMHZoNDhwYjdybTYyOTliIn0.KJyzHWVzu2U087Ps215_LA'

import pandas as pd
import numpy as np
import shapefile
import matplotlib as mplt

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt

#%% user-defined functions
def haversine(lon0,lon1,lat0,lat1):
    r = 6371000.#m                                                                                                                                                                                                                                                 
    lon0 = lon0*np.pi/180
    lon1 = lon1*np.pi/180
    lat0 = lat0*np.pi/180
    lat1 = lat1*np.pi/180
    
    return 2*r*np.arcsin(np.sqrt(np.sin((lat1 - lat0)/2.)**2 +\
		 np.cos(lat0)*np.cos(lat1)*np.sin((lon1 - lon0)/2.)**2))

def mk_map(ax):
    ax.patch.set_visible(False)
    reader = shpreader.Reader('CA_counties/CA_Counties_TIGER2016.shp')
    counties = list(reader.geometries())
    shape_proj = ccrs.epsg(3857)
    COUNTIES = cfeature.ShapelyFeature(counties, shape_proj)
    ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='gray',alpha=0.5)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black',alpha=0.5)
    ax.coastlines('50m')

#%% load data
sensor_list_df1 = pd.read_csv(sensor_list_fn)
moved_IDs = pd.read_csv(moved_IDs_fn)

#%% print number of all indoor and outdoor monitors for paper ("number available for download")
iinds_all = np.where(sensor_list_df1['device_locationtype']==1.0)[0]
oinds_all = np.where(sensor_list_df1['device_locationtype']==0.0)[0]
# only counts A sensors because B's don't have location type copied
print(len(iinds_all),' total inside monitors')
print(len(oinds_all),' total outside monitors') 

# double check B sensors don't have location type copied
iind_count = 0
oind_count = 0
for i in range(sensor_list_df1.shape[0]):
    if sensor_list_df1['parent_id'].iloc[i] == sensor_list_df1['id'].iloc[i]:
        if sensor_list_df1['device_locationtype'].iloc[i] == 1.0:
            iind_count += 1
        elif sensor_list_df1['device_locationtype'].iloc[i] == 0.0:
            oind_count += 1
        else:
            print('A location error')
    else:
        if np.isfinite(sensor_list_df1['device_locationtype'].iloc[i]):
            print('B location error')
print('these numbers should match:')
print(len(iinds_all),iind_count)
print(len(oinds_all),oind_count)

#%% remove moved monitors from the sensor list
moved_ID_inds = []
for ID in moved_IDs['ID'].values:
    moved_ID_inds.append(np.where(sensor_list_df1['thingspeak_primary_id']==ID)[0][0])
sensor_list_df = sensor_list_df1.drop(moved_ID_inds)
sensor_list_df.reset_index(inplace=True,drop=True)

#%% find co-loated monitors
iinds = np.where(sensor_list_df['device_locationtype']==1.0)[0]
oinds = np.where(sensor_list_df['device_locationtype']==0.0)[0]

co_loc_list_df = pd.DataFrame(data={'in_name':[],'in_AID':[],'in_AID_tp':[],'in_lat':[],'in_lon':[],
                                    'out_name':[],'out_AID':[],'out_AID_tp':[],'out_lat':[],'out_lon':[],
                                    'distance':[]})
count = 0
# loop through indoor monitors and find closest outdoor monitor
for iind in iinds:
    slat = sensor_list_df['lat'].iloc[iind]
    slon = sensor_list_df['lon'].iloc[iind]
    
    dists = haversine(slon,sensor_list_df['lon'].iloc[oinds].values,
                      slat,sensor_list_df['lat'].iloc[oinds].values)
    
    if np.nanmin(dists) < max_dist:
        mind = np.where(dists==np.nanmin(dists))[0][0]
        oind = oinds[mind]
        # add to new 'co-loc' data frame 
        # (note this will just be A sensor, will get B later)
        # create df for monitor pair
        mon_df = pd.DataFrame(data={'in_name':[sensor_list_df['label'].iloc[iind]],
                                    'in_AID':[sensor_list_df['id'].iloc[iind]],
                                    'in_AID_tp':[sensor_list_df['thingspeak_primary_id'].iloc[iind]],
                                    'in_AID_tp_key':[sensor_list_df['thingspeak_primary_id_read_key'].iloc[iind]],
                                    'in_lat':[sensor_list_df['lat'].iloc[iind]],
                                    'in_lon':[sensor_list_df['lon'].iloc[iind]],
                                    'out_name':[sensor_list_df['label'].iloc[oind]],
                                    'out_AID':[sensor_list_df['id'].iloc[oind]],
                                    'out_AID_tp':[sensor_list_df['thingspeak_primary_id'].iloc[oind]],
                                    'out_AID_tp_key':[sensor_list_df['thingspeak_primary_id_read_key'].iloc[oind]],
                                    'out_lat':[sensor_list_df['lat'].iloc[oind]],
                                    'out_lon':[sensor_list_df['lon'].iloc[oind]],
                                    'in_mon_type':[sensor_list_df['type'].iloc[iind]],
                                    'out_mon_type':[sensor_list_df['type'].iloc[oind]],
                                    'distance':[np.nanmin(dists)]})
        # append to full list
        co_loc_list_df=co_loc_list_df.append(mon_df,ignore_index='True')        
        count += 1

print(count,' in/out monitors within ',max_dist,' meters')
n_unq_out = len(co_loc_list_df['out_AID'].unique())
n_unq_in = len(co_loc_list_df['in_AID'].unique())
print('n unique in',n_unq_in,'n unique out',n_unq_out)

#%% for these co-located monitors, find B sensor IDs and add to the dataframe
# B sensors have a label 'parent ID' which corresponds to the A sensor ID
in_BIDs = []
out_BIDs = []
in_BIDs_tp = []
out_BIDs_tp = []
in_BIDs_tp_key = []
out_BIDs_tp_key = []
for coind in range(co_loc_list_df.shape[0]):
    in_AID = co_loc_list_df['in_AID'].iloc[coind]
    out_AID = co_loc_list_df['out_AID'].iloc[coind]
    
    # if either montior type is 'unknown' flag to remove
    if co_loc_list_df['in_mon_type'].iloc[coind] =='UNKNOWN':
            in_BIDs.append(-888.0)
            in_BIDs_tp.append(-888.0)
            in_BIDs_tp_key.append(-888.0)
            out_BIDs.append(-888.0)
            out_BIDs_tp.append(-888.0)
            out_BIDs_tp_key.append(-888.0)
            continue
        
    if co_loc_list_df['out_mon_type'].iloc[coind] =='UNKNOWN':
            in_BIDs.append(-888.0)
            in_BIDs_tp.append(-888.0)
            in_BIDs_tp_key.append(-888.0)
            out_BIDs.append(-888.0)
            out_BIDs_tp.append(-888.0)
            out_BIDs_tp_key.append(-888.0)
            continue

    # find BIDs
    # if the monitor is PA-1, no b sensor
    if co_loc_list_df['in_mon_type'].iloc[coind]=='PA-I':
        in_BIDs.append(-555.0)
        in_BIDs_tp.append(-555.0)
        in_BIDs_tp_key.append(-555.0)
    else:
        in_Binds = np.where(sensor_list_df['parent_id'].values==in_AID)[0]
        in_Bind = in_Binds[np.where(sensor_list_df['id'].iloc[in_Binds].values!=in_AID)[0]]
        in_BID = sensor_list_df['id'].iloc[in_Bind].values[0]
        in_BID_tp = sensor_list_df['thingspeak_primary_id'].iloc[in_Bind].values[0]
        in_BID_tp_key = sensor_list_df['thingspeak_primary_id_read_key'].iloc[in_Bind].values[0]

        in_BIDs.append(in_BID)
        in_BIDs_tp.append(in_BID_tp)
        in_BIDs_tp_key.append(in_BID_tp_key)
    
    # if inside monitor is outside, flag
    if co_loc_list_df['out_mon_type'].iloc[coind]=='PA-I':
        out_BIDs.append(-777.0)
        out_BIDs_tp.append(-777.0)
        out_BIDs_tp_key.append(-777.0)
    else:
        out_Binds = np.where(sensor_list_df['parent_id'].values==out_AID)[0]
        out_Bind = out_Binds[np.where(sensor_list_df['id'].iloc[out_Binds].values!=out_AID)[0]]
        out_BID = sensor_list_df['id'].iloc[out_Bind].values[0]
        out_BID_tp = sensor_list_df['thingspeak_primary_id'].iloc[out_Bind].values[0]
        out_BID_tp_key = sensor_list_df['thingspeak_primary_id_read_key'].iloc[out_Bind].values[0]
 
        out_BIDs.append(out_BID)
        out_BIDs_tp.append(out_BID_tp)
        out_BIDs_tp_key.append(out_BID_tp_key)
  
    # use haversine with lat/lons to make sure this works
    if in_BIDs[-1] != -555.0:
        inAlon = co_loc_list_df['in_lon'].iloc[coind]
        inAlat = co_loc_list_df['in_lat'].iloc[coind]
        inBlon = sensor_list_df['lon'].iloc[in_Bind].values
        inBlat = sensor_list_df['lat'].iloc[in_Bind].values
    
        dist = haversine(inAlon,inBlon,inAlat,inAlat)
        if dist > 0:
            print('error. inside A-B distance: ',dist,' m')
    if out_BIDs[-1] != -777.0:
        outAlon = co_loc_list_df['out_lon'].iloc[coind]
        outAlat = co_loc_list_df['out_lat'].iloc[coind]
        outBlon = sensor_list_df['lon'].iloc[out_Bind].values
        outBlat = sensor_list_df['lat'].iloc[out_Bind].values
    
        dist = haversine(outAlon,outBlon,outAlat,outAlat)
        if dist > 0:
            print('error. outside A-B distance: ',dist,' m')

# add to co-loc sensor list
co_loc_list_df['in_BID'] = np.array(in_BIDs)
co_loc_list_df['out_BID'] = np.array(out_BIDs)
co_loc_list_df['in_BID_tp'] = np.array(in_BIDs_tp)
co_loc_list_df['out_BID_tp'] = np.array(out_BIDs_tp)
co_loc_list_df['in_BID_tp_key'] = np.array(in_BIDs_tp_key)
co_loc_list_df['out_BID_tp_key'] = np.array(out_BIDs_tp_key)

#%% count and plot monitors in the western US
lat_max = 49
lat_min = 30
lon_max = -100
lon_min = -130

wUS_inds_A = np.where(np.logical_and(co_loc_list_df['out_lon']<lon_max,
                                     co_loc_list_df['out_lon']>lon_min))[0]
wUS_inds_B = np.where(np.logical_and(co_loc_list_df['out_lat'].iloc[wUS_inds_A]<lat_max,
                            co_loc_list_df['out_lat'].iloc[wUS_inds_A]>lat_min))[0]
wUS_inds = wUS_inds_A[wUS_inds_B]
print('n wUS paris', len(wUS_inds))

# plot to check
fig = go.Figure()
fig.add_trace(go.Scattermapbox(lat=co_loc_list_df['out_lat'],lon=co_loc_list_df['out_lon'],
        mode='markers',hovertext=co_loc_list_df['out_name'],
        marker=go.scattermapbox.Marker(size=9,color='green'),name='outWUS'))
fig.add_trace(go.Scattermapbox(lat=co_loc_list_df['out_lat'].iloc[wUS_inds],lon=co_loc_list_df['out_lon'].iloc[wUS_inds],
        mode='markers',hovertext=co_loc_list_df['out_name'].iloc[wUS_inds],
        marker=go.scattermapbox.Marker(size=9,color='blue'),name='inWUS'))

fig.update_layout(title = str(len(wUS_inds))+' in/out monitors in the western US',
    autosize=True,hovermode='closest',mapbox=dict(accesstoken=mapbox_access_token,
        bearing=0,center=dict(lat=38.92,lon=-77.07),pitch=0,zoom=4),)
fig.write_html(out_fig_path + 'inoutmap_maxd'+str(max_dist)+'_wUS.html')
fig.show()

#%% create flag by select county 
cnty_file = shapefile.Reader(shapefile_fn_county)
cnty_records = cnty_file.records()
cnty_shapes = cnty_file.shapes()
si = 0
county_flag = np.array(['00000000000000000']*co_loc_list_df.shape[0])
FIPS_flag = np.array(['00000']*co_loc_list_df.shape[0])
area_abbr = np.array(['000']*co_loc_list_df.shape[0])

for j in range(len(cnty_records)):
        #name = cnty_records[j][5]
        #FIPs = cnty_records[j][4]
        FIPs = cnty_records[j][4]
        name = cnty_records[j][3]
        # note there are multiple counties with some of these names, but this is
        # just to set area codes, we actually select monitors by their FIPs code,
        # which is more precise.
        if name in ['King','Multnomah','Clackamas']:
            abbr = 'PNW'
        elif name in ['Los Angeles','Ventura','San Bernardino','Orange','Riverside']:
            abbr = 'LA'
        elif name in ['San Francisco','San Mateo','Sonoma','Napa','Solano','Contra Costa','Alameda','Santa Clara','Marin']:
            abbr = 'SF'
        elif name in ['Salt Lake']:
            abbr = 'SLC'
        elif name in ['Larimer','Boulder','Denver','Weld','Jefferson',
                    'Adams','Arapahoe','Broomfield','Douglas','El Paso','Elbert',
                    'Teller','Pueblo']:
            abbr = 'CFR'
        else:
            abbr = 'NA' # should never assign this, a good check to make sure this is working
        '''
        this list of counties matches the fips codes below
        if name in ['King','San Francisco','Los Angeles','Ventura','San Bernardino','Orange','Riverside',
                    'San Mateo','Sonoma','Napa','Solano','Contra Costa','Alameda','Santa Clara','Marin',
                    'Multnomah','Clackamas','Salt Lake', 
                    'Larimer','Denver','Weld','Boulder',
                    'Adams','Arapahoe','Jefferson','Broomfield','Douglas','El Paso',
                    'Elbert','Teller','Pueblo']:
        '''
        # look for monitors in the counties we are interested in
        if FIPs in ['53033','06075','06037','06111','06071','06059','06065',
                    '06081','06097','06055','06095','06013','06001','06085','06041',
                    '41051','41005','49035',
                    '08069','08031','08123','08013',
                    '08001','08005','08059','08014','08035','08041',
                    '08039','08119','08101']:
            #print(name)
            cnty_shp = cnty_shapes[j]
            for i in range(len(cnty_shp.parts)):
                i0 = cnty_shp.parts[i]
                if i < len(cnty_shp.parts)-1:
                		i1 = cnty_shp.parts[i+1] - 1
                else:
                		i1 = len(cnty_shp.points)
                seg = cnty_shp.points[i0:i1+1]
                mpath = mplt.path.Path(seg)
                # flag for list
                points = np.array((co_loc_list_df['in_lon'].values,
                                   co_loc_list_df['in_lat'].values)).T
                mask = mpath.contains_points(points)#.reshape(glon.shape)
                county_inds = np.where(mask==True)[0]
                county_flag[county_inds] = name
                FIPS_flag[county_inds] = FIPs
                area_abbr[county_inds] = abbr

# check that we never assign 'NA' area name
print('check that "NA" is NOT in this list:',np.unique(area_abbr))
    
# add flag to co_loc_list
co_loc_list_df['county_name'] = county_flag
co_loc_list_df['FIPS_code'] = FIPS_flag
co_loc_list_df['area_abbr'] = area_abbr

#%% write colocated list to file
# write full list to file
co_loc_list_df.to_csv(out_fp + 'co_loc_sensor_list_'+str(int(max_dist))+'m_'+out_desc+'.csv')

# write westUS list to file
wUS_co_loc_list_df = co_loc_list_df.iloc[wUS_inds]
wUS_co_loc_list_df.reset_index(inplace=True,drop=True)
wUS_co_loc_list_df.to_csv(out_fp + 'wUS_co_loc_sensor_list_'+str(int(max_dist))+'m_'+out_desc+'.csv')

# now write abbreviated list
rmv_inds = np.where(FIPS_flag == '00000')[0]
co_loc_list_df_scounties = co_loc_list_df.drop(rmv_inds)
co_loc_list_df_scounties.reset_index(inplace=True,drop=True)
co_loc_list_df_scounties.to_csv(out_fp + 'co_loc_sensor_list_select_counties_'+str(int(max_dist))+'m_'+out_desc+'.csv')

print(len(np.where(FIPS_flag!='00000')[0]),'monitors saved in select counties')

#%% write list of thingspeak IDs and keys for Bonne
# first remove co-loc-list rows with -777 and -888 (keep -555, these are indoor monitors without a B sensor)
rmv_indsA = np.where(co_loc_list_df['out_BID']<-700.0)[0]
rmv_indsB = np.where(co_loc_list_df['in_BID']<-700.0)[0]
rmv_inds = np.unique(np.hstack([rmv_indsA,rmv_indsB]))

co_loc_list_Bonne = co_loc_list_df.drop(rmv_inds)

all_tp_ids = np.hstack([co_loc_list_Bonne['out_AID_tp'].values,
                        co_loc_list_Bonne['in_AID_tp'].values,
                        co_loc_list_Bonne['out_BID_tp'].values,
                        co_loc_list_Bonne['in_BID_tp'].values])
all_tp_keys = np.hstack([co_loc_list_Bonne['out_AID_tp_key'].values,
                        co_loc_list_Bonne['in_AID_tp_key'].values,
                        co_loc_list_Bonne['out_BID_tp_key'].values,
                        co_loc_list_Bonne['in_BID_tp_key'].values])
all_lat = np.hstack([co_loc_list_Bonne['out_lat'].values,
                        co_loc_list_Bonne['in_lat'].values,
                        co_loc_list_Bonne['out_lat'].values,
                        co_loc_list_Bonne['in_lat'].values])
all_lon = np.hstack([co_loc_list_Bonne['out_lon'].values,
                        co_loc_list_Bonne['in_lon'].values,
                        co_loc_list_Bonne['out_lon'].values,
                        co_loc_list_Bonne['in_lon'].values])
all_fips = np.hstack([co_loc_list_Bonne['FIPS_code'].values,
                        co_loc_list_Bonne['FIPS_code'].values,
                        co_loc_list_Bonne['FIPS_code'].values,
                        co_loc_list_Bonne['FIPS_code'].values])

nrows = co_loc_list_Bonne.shape[0]
all_types = np.hstack([['out_A']*nrows,['in_A']*nrows,['out_B']*nrows,['in_B']*nrows])

rmvinds = np.where(all_tp_ids == -555.0)[0]
all_ids_4bonne = pd.DataFrame(data={'thingspeak_primary_id':all_tp_ids,
                                    'thingspeak_primary_key':all_tp_keys,
                                    'lat':all_lat,
                                    'lon':all_lon,
                                    'co_loc_mon_type':all_types,
                                    'fips_code':all_fips})
all_ids_4Bonne_save = all_ids_4bonne.drop(rmvinds)

# drop duplicates - duplicate outdoor monitors paired to same indoor monitors
all_ids_4Bonne_save.drop_duplicates(ignore_index=True,inplace=True)
all_ids_4Bonne_save.reset_index(inplace=True,drop=True)
all_ids_4Bonne_save.to_csv(out_fp + 'co_loc_sensor_list_d'+str(int(max_dist))+'m_'+out_desc+'_4Bonne.csv')

# now drop monitors outside select counties
FIPSselect_ids_4Bonne_save = all_ids_4Bonne_save.drop(np.where(all_ids_4Bonne_save['fips_code']=='00000')[0])
FIPSselect_ids_4Bonne_save.reset_index(inplace=True,drop=True)
FIPSselect_ids_4Bonne_save.to_csv(out_fp + 'co_loc_sensor_list_d'+str(int(max_dist))+'m_'+out_desc+'_4Bonne_FIPSselect.csv')

print('data saved, making figures')

#%% plot co-located monitors
fig = go.Figure()
fig.add_trace(go.Scattermapbox(
        lat=co_loc_list_df['in_lat'],
        lon=co_loc_list_df['in_lon'],
        mode='markers',hovertext=co_loc_list_df['in_name'],
        marker=go.scattermapbox.Marker(size=9,color='green'),name='inside'))
fig.add_trace(go.Scattermapbox(
        lat=co_loc_list_df['out_lat'],
        lon=co_loc_list_df['out_lon'],
        mode='markers',hovertext=co_loc_list_df['out_name'],
        marker=go.scattermapbox.Marker(size=9,color='blue'),name='outside'))
fig.update_layout(title = str(count)+' in/out monitors within '+str(max_dist)+'m',
    autosize=True,hovermode='closest',
    mapbox=dict(accesstoken=mapbox_access_token,bearing=0,
        center=dict(lat=38.92,lon=-77.07),pitch=0,zoom=4),)
fig.write_html(out_fig_path + 'inoutmap_maxd'+str(max_dist)+'.html')
fig.show()

# check state/county assignment
fig = go.Figure()
# plot in loop so the different fips codes show up as different colors
for c in all_ids_4Bonne_save['fips_code'].unique():
    inds = np.where(all_ids_4Bonne_save['fips_code']==c)
    fig.add_trace(go.Scattermapbox(
            lat=all_ids_4Bonne_save['lat'].iloc[inds],
            lon=all_ids_4Bonne_save['lon'].iloc[inds],
            mode='markers',
            marker=go.scattermapbox.Marker(size=9),name=c))
fig.update_layout(title = 'check FIPS assignment',
    autosize=True,hovermode='closest',
    mapbox=dict(accesstoken=mapbox_access_token,bearing=0,
        center=dict(lat=38.92,lon=-77.07),pitch=0,zoom=4),)
fig.write_html(out_fig_path + 'county-assignment'+str(max_dist)+'.html')
fig.show()

#%% plot shapefile with monitors to confirm assignment
fig,axarr = plt.subplots(ncols=1,nrows=1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
i=0
ax = axarr
mk_map(ax)
ax.outline_patch.set_edgecolor('white')
for c in all_ids_4Bonne_save['fips_code'].unique():
    if c=='00000':
        continue
    inds = np.where(all_ids_4Bonne_save['fips_code']==c)
    cs = ax.scatter(all_ids_4Bonne_save['lon'].iloc[inds],
                    all_ids_4Bonne_save['lat'].iloc[inds], i,
                    transform=ccrs.PlateCarree(),cmap='magma',vmin=0,vmax=15.0)
    i+=1
plt.tight_layout()
plt.show()

#%% make subset of monitors in california for prioritizing downloads
'''
#shapefile_fn_state = '/Users/kodell/Local Google Drive /CSU/Research/NSF/smoke-specific HIA/smoke-specific HIA scripts/cb_2018_us_state_20m/cb_2018_us_state_20m'
shapefile_fn_state = '/Users/kodell/Local Google Drive /CSU/Research/NSF/purple_air_smoke/PA_data/cb_2018_us_state_500k/cb_2018_us_state_500k'

states_file = shapefile.Reader(shapefile_fn_state)
states_shp = states_file.shape(0)
state_records = states_file.records()
state_shapes = states_file.shapes()
si = 0
in_cal = np.array([False]*all_ids_4Bonne_save.shape[0])

for j in range(len(state_records)):
        name = state_records[j][4]
        if name in ['CA']:
            print(name)
            plume_shp = state_shapes[j]
            for i in range(len(plume_shp.parts)):
                i0 = plume_shp.parts[i]
                if i < len(plume_shp.parts)-1:
                		i1 = plume_shp.parts[i+1] - 1
                else:
                		i1 = len(plume_shp.points)
                seg = plume_shp.points[i0:i1+1]
                mpath = mplt.path.Path(seg)
                points = np.array((all_ids_4Bonne_save['lon'].values,
                                   all_ids_4Bonne_save['lat'].values)).T
                mask = mpath.contains_points(points)#.reshape(glon.shape)
                CA_inds = np.where(mask==True)[0]
                in_cal[CA_inds] = True

CArmv_inds = np.where(~(in_cal))[0]
all_ids_4Bonne_save_CA = all_ids_4Bonne_save.drop(CArmv_inds)
all_ids_4Bonne_save_CA.drop_duplicates(ignore_index=True,inplace=True)
all_ids_4Bonne_save_CA.reset_index(inplace=True,drop=True)
all_ids_4Bonne_save_CA.to_csv(out_fp + 'co_loc_sensor_list_d'+str(int(max_dist))+'m_'+out_desc+'_4Bonne_CA.csv')
'''