from Utility import *

raise ValueError('To stop running')

bmeta = pd.read_feather(dataDir/'building_meta.feather', use_threads=True)
bmeta2 = bmeta.set_index(['site_id', 'building_id'])


tr_meter = pd.read_feather(dataDir / 'train.feather', use_threads=True)

# tr_meter.set_index('building_id')

tr_meter['meter_reading'] = np.log1p(tr_meter['meter_reading'])
# tr_meter.set_index('timestamp', inplace=True)
tr_meter = tr_meter.merge(bmeta[['site_id','building_id']], on='building_id', how='left')
tr_meter.set_index('timestamp',inplace=True)

bid_sid_indexer = bmeta[['site_id','building_id']].set_index('building_id')
# site_bd_range = bmeta.groupby('site_id').building_id.agg([min, max]).add_prefix('bid_')


mt_gps = tr_meter.groupby('meter')
meter0 = mt_gps.get_group(0).set_index('site_id',append=True).swaplevel()

meter0.reset_index().set_index(['timestamp','building_id']).index.duplicated().sum()


# for training data
meter0_cleaned_1, meter0_mask1 = clean_meter_reading2(meter0.drop(8, level='site_id').droplevel('site_id'))
meter0_cleaned_2, meter0_mask2 = clean_meter_reading2(meter0.loc[8], max_tol=72)
meter0_cleaned_mat = pd.concat([meter0_cleaned_1,meter0_cleaned_2], axis=1)


meter1_cleaned_mat, meter1_mask = clean_meter_reading2(mt_gps.get_group(1), max_tol=48, drop_zero=False)
meter2_cleaned_mat, meter2_mask = clean_meter_reading2(mt_gps.get_group(2), max_tol=48, drop_zero=False)
meter3_cleaned_mat, meter3_mask = clean_meter_reading2(mt_gps.get_group(3), max_tol=48, drop_zero=False)

meter1_cleaned_mat2, _ = clean_temporal_bands(meter1_cleaned_mat, bmeta, thresh=0.1, site_thresh_dict={9:0.3})
meter2_cleaned_mat2, _ = clean_temporal_bands(meter2_cleaned_mat, bmeta, thresh=0.3)
meter3_cleaned_mat2, _ = clean_temporal_bands(meter3_cleaned_mat, bmeta, thresh=0.3)
# """
# Resuslt visualization for clean_temporal_bands
# """
# plt.figure()
# sns.heatmap(meter2_cleaned_mat)
#
# plt.figure()
# sns.heatmap(meter2_cleaned_mat2)

meter1_cleaned_mat3, _ = clean_isolated_bands(meter1_cleaned_mat2,)
meter2_cleaned_mat3, _ = clean_isolated_bands(meter2_cleaned_mat2,)
meter3_cleaned_mat3, _ = clean_isolated_bands(meter3_cleaned_mat2, exempt_sites=[1])



# """
# Resuslt visualization for clean_isolated_bands
# """
# fig, axes = plt.subplots(1,3, sharey=True,) # sharex=True)
# # sns.heatmap(nonNA_ratio)
# axes[0].imshow(meter_mat, aspect='auto', )
# axes[1].imshow(nonNA_ratio, aspect='auto',vmax=0.3,)
# axes[2].imshow(isolated_bands,aspect='auto')
# axes[2].spy(isolated_bands,aspect='auto')


reformat_meter0 = reformat_meter_mat(meter0_cleaned_mat, bmeta, meter_type=0)
reformat_meter1 = reformat_meter_mat(meter1_cleaned_mat3, bmeta, meter_type=1)
reformat_meter2 = reformat_meter_mat(meter2_cleaned_mat3, bmeta, meter_type=2)
reformat_meter3 = reformat_meter_mat(meter3_cleaned_mat3, bmeta, meter_type=3)



cleaned_meters = pd.concat([reformat_meter0,reformat_meter1,reformat_meter2,reformat_meter3],axis=0).sort_values(['timestamp', 'building_id'])
assert cleaned_meters.meter_reading.isna().sum() == 0


cleaned_meters_copy = cleaned_meters.copy().set_index('timestamp')
cleaned_meters_copy_gps = cleaned_meters_copy.groupby('meter')

mtype = 3
fig, axes = plt.subplots(1,2, sharey=True, sharex=True)
sns.heatmap(get_meter_mat_copy(mt_gps.get_group(mtype)), ax=axes[0])
axes[0].set_title('original')
sns.heatmap(get_meter_mat_copy(cleaned_meters_copy_gps.get_group(mtype)), ax=axes[1])
axes[1].set_title('post cleaning')
plt.suptitle(f'meter {mtype}')

plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1,2, sharey=True, sharex=True)
axes[0].imshow(get_meter_mat_copy(mt_gps.get_group(mtype)), aspect='auto')
axes[0].set_title('original')
axes[1].imshow(get_meter_mat_copy(cleaned_meters_copy_gps.get_group(mtype)), aspect='auto')
axes[1].set_title('post cleaning')
plt.suptitle(f'meter {mtype}')
raise ValueError('To stop running')

# todo add checks before saving
cleaned_meters.reset_index(drop=True).to_feather(processed_data_dir/'cleaned_tr_meter_log_1217.feather')

raise ValueError('To stop running')
# cleaned_meters = pd.read_feather(dataDir/'cleaned_tr_meter_logarithm_v3.2.feather', use_threads=True)



cleaned_meters.set_index('timestamp', inplace=True)

# cleaned_meters = tr_meter
cleaned_meters['time_delta'] = (cleaned_meters.index.get_level_values(level='timestamp') - pd.to_datetime("2016-01-01")).total_seconds() // 3600
cleaned_meters['time_delta'] = cleaned_meters['time_delta'].astype(int, copy=False)




cleaned_meters.reset_index(inplace=True)
bad_zeros = find_bad_zeros(cleaned_meters)

bad_bid1099 = find_bad_building1099(cleaned_meters)

all_bad_zeros = bad_zeros.union(bad_bid1099)
cleaned_meters2 = cleaned_meters.drop(cleaned_meters.index[bad_zeros])
cleaned_meters2.drop(columns=['time_delta'],inplace=True)
cleaned_meters2 = cleaned_meters2.reset_index(drop=True)
assert cleaned_meters2.meter_reading.isna().sum() == 0
# todo add checks before saving

cleaned_meters2.to_feather(processed_data_dir/'cleaned_tr_meter_log_1219.feather')
# cleaned_meters.reset_index(drop=True).to_feather(processed_data_dir/'cleaned_leak_logarithm_v3.2.2.feather')


cmt2_gp = cleaned_meters2.groupby('meter')
cmeter = cmt2_gp.get_group(0).set_index(['timestamp','building_id'])['meter_reading'].sort_index()
cmeter.isna().sum()
cmeter.index.duplicated().sum()
cmeter[cmeter.index.duplicated(keep=False)]
""" 
############################################################################################################
#########################################  WEATHER DATA  ##############################################
############################################################################################################ """

all_wthr = pd.read_feather(dataDir / 'all_raw_wthr_n_solar_localtime.feather', use_threads=True)


wthr_max_gap = 6
lag_interval = 6

wthr_cleaned = preprocess_weather_data(all_wthr, wthr_max_gap=wthr_max_gap)


"""
===========  check if weather aligned  ==================
"""
# del all_wthr,wthr_cleaned,check_wthr

# check_wthr = append_time_feature(wthr_cleaned)
# check_wthr = convert_UTC_to_LocalTime(all_wthr)
# check_wthr = append_time_feature(check_wthr.set_index('timestamp'))
# check_wthr = convert_UTC_to_LocalTime(wthr_cleaned.reset_index())
check_wthr = wthr_cleaned.reset_index().copy()
check_wthr = append_time_feature(check_wthr.set_index('timestamp'))
mean_check_wthr = check_wthr.groupby(['site_id','hour'])['air_temperature'].mean()
mean_check_wthr = mean_check_wthr.reset_index()

# plt.figure()
sns.relplot(x='hour', y='air_temperature', hue='site_id',kind='line',data=mean_check_wthr)

"""
===========  end of check  ==================
"""

def prepare_wthr_to_save(wthr):
    wthr[['wind_dir_sin', 'wind_dir_cos']] = wthr[['wind_dir_sin', 'wind_dir_cos']].round(decimals=2)
    # wthr.reset_index('site_id',inplace=True)
    # # reduce_mem_usage(wthr)  # compress (optional)
    # wthr.reset_index(inplace=True)


prepare_wthr_to_save(wthr_cleaned)
# wthr_cleaned.reset_index().to_feather(dataDir/'cleaned_all_wthr_v1.feather')

















tst_meter_bmeta_wthr = pd.read_feather(inputDir/'tst_data_w_feature.feather', use_threads=True)
fbase_name = 'tst_data_w_feature'

n_chuncks = 10
chunk_size, residuals = divmod(tst_meter_bmeta_wthr.shape[0], n_chuncks)

iloc_start = 0
iloc_stop = 0
for i in range(n_chuncks):
    iloc_stop = iloc_start + chunk_size
    if i == n_chuncks-1:
        iloc_stop += residuals
    print(iloc_start)
    print(iloc_stop)
    print(tst_meter_bmeta_wthr.iloc[iloc_start:iloc_stop, :].shape)
    tst_meter_bmeta_wthr.iloc[iloc_start:iloc_stop,:].reset_index(drop=True).to_feather(inputDir/f'{fbase_name}_chunck{i}.feather')
    print()
    iloc_start = iloc_stop





























