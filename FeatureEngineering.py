from Utility import *

raise ValueError('To stop running')

lb_encoder_dict = {}

bmeta = pd.read_feather(dataDir/'building_meta.feather', use_threads=True)
"""
'site_id', 'building_id' are already ints starting from 0, no need for encoding
"""
bmeta_cats_to_encode = ['primary_use', 'year_built',]# 'floor_count']
bmeta[bmeta_cats_to_encode] = bmeta[bmeta_cats_to_encode].apply(get_label_encoders, args=(lb_encoder_dict,))



""" #########################################  WEATHER DATA  ##############################################
############################################################################################################ """


wthr_cleaned = pd.read_feather(dataDir/'cleaned_all_wthr_v1.feather', use_threads=True).set_index(['site_id', 'timestamp'])


features_for_lag = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'radiation', 'wind_speed', 'cloud_coverage']


# lag_fts = add_time_lag_diff_feature2(wthr_cleaned, features_for_lag, lag_interval=2, total_span=8).sort_index()
# from sklearn import preprocessing
# from sklearn.decomposition import PCA
# pca = PCA(n_components=4,whiten=True)
# # lag_data = lag_site.filter(regex='air_temperature_diff_lag_.*').dropna().T.sort_index().T
# lag_data = lag_site.filter(regex='radiation_diff_lag_.*').dropna().T.sort_index().T
# pca_lag = pca.fit_transform(preprocessing.scale(lag_data))
#
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter3D(pca_lag[:,0], pca_lag[:,1], pca_lag[:,2],s=1,c=lag_data.index.get_level_values('timestamp'),cmap='jet')
# ax.plot(pca_lag[:,0], pca_lag[:,1], pca_lag[:,2],c='k',lw=0.5,alpha=0.3)
# pca.explained_variance_ratio_

from sklearn import preprocessing
from sklearn.decomposition import PCA
pca = PCA(whiten=True)
data_to_reduce = wthr_cleaned[['air_temperature','dew_temperature']].dropna()
air_dew_pca = pca.fit_transform(preprocessing.scale(data_to_reduce))
air_dew_pca_df = pd.DataFrame(air_dew_pca, index=data_to_reduce.index, columns=['air_dew_pc1','air_dew_pc2'])
pca.explained_variance_ratio_

# plt.figure()
# plt.plot(air_dew_pca[:1000,0],air_dew_pca[:1000,1],lw=0.5)
# plt.plot(wthr_cleaned['air_temperature'],wthr_cleaned['dew_temperature'],lw=0.5)

data_to_reduce2 = wthr_cleaned[['air_temperature','dew_temperature','radiation']].dropna()
air_dew_rad_pca = pca.fit_transform(preprocessing.scale(data_to_reduce2))
air_dew_rad_pca_df = pd.DataFrame(air_dew_rad_pca, index=data_to_reduce2.index, columns=['air_dew_rad_pc1','air_dew_rad_pc2','air_dew_rad_pc3'])

# plt.figure()
# plt.plot(air_dew_rad_pca[:1000,0],air_dew_rad_pca[:1000,1],lw=0.5)

lag_pca1 = get_lag_pca(wthr_cleaned, features_for_lag, lag_interval=2, total_span=8)
lag_pca2 = get_lag_pca(wthr_cleaned, features_for_lag, lag_interval=2, total_span=16)
lag_pca3 = get_lag_pca(wthr_cleaned, features_for_lag, lag_interval=3, total_span=48)

final_wthr = pd.concat([wthr_cleaned,lag_pca1,lag_pca2,lag_pca3,air_dew_pca_df,air_dew_rad_pca_df], axis=1)
final_wthr['daylight_saving'] = get_label_encoders(final_wthr['daylight_saving'], lb_encoder_dict)

# final_wthr.drop(columns='daylight_saving')

final_wthr.reset_index().to_feather(inputDir/'final_wthr_1217.feather')



"""    
######################  Combine TRAIN METER_READING with building & weather meta  ################################
############################################################################################################ 
"""
final_wthr = pd.read_feather(inputDir/'final_wthr_1217.feather')
cleaned_meters = pd.read_feather(processed_data_dir/'cleaned_tr_meter_log_1219.feather', use_threads=True)
append_time_feature2(cleaned_meters)

cleaned_meters.set_index('timestamp', inplace=True)
get_holidays(cleaned_meters)
cleaned_meters.reset_index(inplace=True)


# final_wthr = pd.read_feather(inputDir/'final_wthr_1217.feather', use_threads=True)
comb_all_meters = cleaned_meters.merge(final_wthr, on=['site_id', 'timestamp'], how='left')
comb_all_meters = comb_all_meters.merge(bmeta, on=['site_id', 'building_id'], how='left')

# creat_features(comb_all_meters)

comb_all_meters.to_feather(inputDir/'train_data_w_feature_1219.feather')

import pickle
with open(inputDir/'label_encoders_1217_v2.lb','wb') as f:
    pickle.dump(lb_encoder_dict, f)


"""    
######################  Combine LEAK METER_READING with building & weather meta  ################################
############################################################################################################ 
"""

leak_meters = pd.read_feather(processed_data_dir/'cleaned_leak_meter_log_1219.feather', use_threads=True)
append_time_feature2(leak_meters)

leak_meters.set_index('timestamp', inplace=True)
get_holidays(leak_meters)
leak_meters.reset_index(inplace=True)

comb_leak = leak_meters.merge(final_wthr, on=['site_id', 'timestamp'], how='left')
comb_leak = comb_leak.merge(bmeta, on=['site_id', 'building_id'], how='left')

comb_leak.to_feather(inputDir/'leak_data_w_feature_1219.feather')

raise ValueError('To stop running')


"""    
######################  Combine TEST with building & weather meta  ################################
############################################################################################################ 
"""
import pickle
with open(inputDir/'label_encoders_1217.lb','rb') as f:
    lb_encoder_dict = pickle.load(f)

bmeta = pd.read_feather(dataDir/'building_meta.feather', use_threads=True)

final_wthr = pd.read_feather(inputDir/'final_wthr_1217.feather')
tst_meter = pd.read_feather(dataDir/'tst_meter.feather', use_threads=True)
append_time_feature2(tst_meter)

comb_tst = tst_meter.merge(bmeta, on=['building_id'], how='left')
del tst_meter
comb_tst.set_index('timestamp', inplace=True)
get_holidays(comb_tst)
comb_tst.reset_index(inplace=True)
comb_tst = comb_tst.merge(final_wthr, on=['site_id', 'timestamp'], how='left')


auto_lb_transform(comb_tst, lb_encoder_dict)

comb_tst.to_feather(inputDir/'tst_data_w_feature_1217.feather')



# comb_tst = pd.read_feather(inputDir/'tst_data_w_feature_1217.feather', use_threads=True)
fbase_name = 'tst_data_w_feature_1217'

n_chuncks = 10
chunk_size, residuals = divmod(comb_tst.shape[0], n_chuncks)

iloc_start = 0
iloc_stop = 0
for i in range(n_chuncks):
    iloc_stop = iloc_start + chunk_size
    if i == n_chuncks-1:
        iloc_stop += residuals
    print(iloc_start)
    print(iloc_stop)
    print(comb_tst.iloc[iloc_start:iloc_stop, :].shape)
    comb_tst.iloc[iloc_start:iloc_stop,:].reset_index(drop=True).to_feather(inputDir/f'{fbase_name}_chunck{i}.feather')
    print()
    iloc_start = iloc_stop











