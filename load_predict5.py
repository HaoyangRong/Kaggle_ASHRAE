from Utility import *
import time
start_time = time.time()

# TODO ADD CHECKER FOR CAT ENCODING
bmeta_cats_to_encode = ['primary_use', 'year_built',]# 'floor_count']
final_cat_fts = ['site_id', 'building_id', 'hour', 'dayofweek', 'is_holiday(new)'] + bmeta_cats_to_encode

feature_set1 = [
    'air_dew_pc1',
    'air_dew_pc2',
    'air_temperature_span8_pc1',
    'air_temperature_span8_pc2',
    'radiation_span8_pc1',
    'radiation_span8_pc2',
    # 'air_dew_rad_pc1',#'air_dew_rad_pc2','air_dew_rad_pc3',
    'site_id',
    'building_id',
    'primary_use',
    'square_feet',
    'year_built',
    'floor_count',
    'cloud_coverage',
    'precip_depth_1_hr',
    'sea_level_pressure',
    'wind_direction',
    'wind_speed',
    'hour',
    'dayofweek',
]
# feature_set1 = basic_feature_set


feature_set2 = ['is_holiday(new)', 'holiday_density']

feature_set = feature_set1 + feature_set2

# set(feature_set).difference(basic_feature_set)
# set(basic_feature_set).difference(feature_set)
print(f'Feature set size: {len(feature_set)}')

print('Preparing train data...')
comb_all_meters = pd.read_feather(inputDir/'train_data_w_feature_1219.feather').set_index('timestamp')
comb_all_meters = comb_all_meters[feature_set1+['meter','meter_reading']]


print('Preparing leak data...')
leak_all_meters = pd.read_feather(inputDir/'leak_data_w_feature_1219.feather', use_threads=True).set_index('timestamp')
leak_all_meters = leak_all_meters['2017':'2018']
leak_all_meters = leak_all_meters[feature_set1+['meter','meter_reading']]
# get_holidays(leak_all_meters)

print('Combining train and leak...')
super_all_meters = pd.concat([comb_all_meters,leak_all_meters], axis=0)
assert super_all_meters.shape[0] == (comb_all_meters.shape[0]+leak_all_meters.shape[0])
super_all_meters = get_holidays2(super_all_meters)
assert set(super_all_meters.columns) == set(feature_set+['meter','meter_reading'])
super_all_meters[final_cat_fts] = super_all_meters[final_cat_fts].astype('category', copy=False)
try:
    del comb_all_meters
    del leak_all_meters
except:
    pass



all_mt_gps = super_all_meters.groupby('meter')

tr_meters = {}
for meter_type, meter in all_mt_gps:
    " remove site_0 readings (optional)"
    # assert meter.index.names[0] == 'site_id'
    tr_meters[meter_type] = meter

print('########################################################################\n'
      'Training models...\n'
      '########################################################################')
model_kwargs = dict(
    objective='regression',
    metric='rmse',
    num_threads=15,
    num_leaves=50,
    # feature_fraction_bynode=0.5,
    bagging_fraction=0.1,
    bagging_freq=1,
    feature_fraction=0.5,
    # cat_smooth=100,
    # learning_rate=0.05
    # subsample_for_bin=10000
)

train_results = {}
models_fts_dtypes = {}
for meter_type, meter in tr_meters.items():
    print(f'Training meter type {meter_type}...')
    period_index, folds = get_month_folds3(tr_meters,mtype=meter_type)

    comb_data = tr_meters[meter_type]
    comb_data = comb_data.set_index(period_index, append=True).swaplevel()
    input_X = comb_data.drop(columns=['meter_reading'])
    target = comb_data[['meter_reading', 'building_id']].set_index('building_id', append=True)

    models_fts_dtypes[meter_type] = input_X.dtypes
    train_results[meter_type] = lgb_train(input_X, target, cv_iterator=folds, model_params=model_kwargs)


raise ValueError('To stop running')

import pickle
with open(train_output_dir/'trained_models.model','wb') as f:
    pickle.dump({'fts_dtypes':models_fts_dtypes,'train_results':train_results},f)

try:
    del tr_meters
    del super_all_meters
except:
    pass

#
# import pickle
# with open(train_output_dir/'trained_models.model','rb') as f:
#     train_results_dict = pickle.load(f)

# train_results = train_results_dict['train_results']
# models_fts_dtypes = train_results_dict['fts_dtypes']

print('Making predictions...')
n_chuncks = 10
for chunck_i in range(n_chuncks):
    print()
    print('******************************************************************')
    print(f'Making predictions on chunck{chunck_i}...')
    # tst_meter_bmeta_wthr = pd.read_feather(inputDir/f'tst_data_w_feature_1217_chunck{chunck_i}.feather', use_threads=True).set_index('row_id')
    tst_meter_bmeta_wthr = pd.read_feather(inputDir/f'tst_data_w_feature_1217_chunck{chunck_i}.feather', use_threads=True)
    # print(tst_meter_bmeta_wthr.sample(10).to_string())
    tst_meter_bmeta_wthr.set_index('timestamp',inplace=True)
    tst_meter_bmeta_wthr = get_holidays2(tst_meter_bmeta_wthr)
    tst_meter_bmeta_wthr.reset_index(inplace=True)
    tst_meter_bmeta_wthr.set_index('row_id', inplace=True)
    meter_gps = tst_meter_bmeta_wthr.groupby('meter')

    """  Make prediction and save to submission  """

    for meter_type, meter_results in train_results.items():
        print('********************************')
        print(f'Predicting meter {meter_type}...')
        continue_rest = True
        if meter_type in meter_gps.groups.keys():
            tst_input = meter_gps.get_group(meter_type)
        else:
            print(f'Skipped meter {meter_type}')
            continue_rest = False

        if continue_rest:
            tst_input = tst_input.loc[:, models_fts_dtypes[meter_type].index]
            assert tst_input.index.name == 'row_id'
            dtypes_tst = tst_input.dtypes.apply(lambda dtype :dtype.name)
            dtypes_tr = models_fts_dtypes[meter_type].apply(lambda dtype: dtype.name)
            assert dtypes_tst.index.to_list() == dtypes_tr.index.to_list()
            model_cat_fts = dtypes_tr.index[dtypes_tr.astype(str) == 'category']
            tst_input[model_cat_fts] = tst_input[model_cat_fts].astype('category', copy=False)

            # assert dtypes_tst.equals(dtypes_tr)


            meter_models = meter_results['models']
            # preds_array = np.zeros((len(meter_models), tst_input.shape[0]))
            for i, (cv_name, model) in enumerate(train_results[meter_type]['models'].items()):
                print(f'Predicting with model {i}')
                # np.save(dataDir/f'pred_meter{meter_type}_model{i}',model.predict(tst_input))
                pd.DataFrame(model.predict(tst_input), index=tst_input.index, columns=['meter_reading']).reset_index().to_feather(test_output_dir/f'pred_meter{meter_type}_model{i}_chunck{chunck_i}.feather')


print('Merging predictions...')
import os.path
tst_preds = {}
for meter_type, meter_results in train_results.items():
    print(f'Loading meter {meter_type} predictions...')
    first_chunk = True
    for chunck_i in range(n_chuncks):
        pred ={}
        for model_i, (cv_name, model) in enumerate(train_results[meter_type]['models'].items()):
            continue_rest = True
            fname = test_output_dir/f'pred_meter{meter_type}_model{model_i}_chunck{chunck_i}.feather'
            if os.path.isfile(fname):
                print(f'pred_meter{meter_type}_model{model_i}_chunck{chunck_i}')
                pred[model_i] = pd.read_feather(fname).set_index('row_id')
                # print('here')
                # if pred[model_i].index.contains(8):
                #     raise ValueError('To stop running')
            else:
                print('skipped')
                continue_rest = False

        if continue_rest:
            print('Averaging predctions...')
            mean_pred = pd.concat(pred.values(), axis=1).mean(axis=1)
            if first_chunk:
                mean_preds = mean_pred
            else:
                mean_preds = pd.concat([mean_preds, mean_pred], axis=0)
            first_chunk = False

    tst_preds[meter_type] = mean_preds
    # raise ValueError('To stop running')


tst_pred_sr = pd.concat(tst_preds.values(), axis=0).sort_index().rename('meter_reading')
final_tst_pred = np.expm1(tst_pred_sr).reset_index().clip(lower=0)

final_tst_pred.to_feather(dataDir/'submission_no_leak.feather')

cleaned_submission = final_tst_pred




print('########################################################################\n'
      'Update submission with leak data\n'
      '########################################################################')



# submission0 = pd.read_csv('submission0.csv')
# dup_idx = submission0['row_id'].duplicated(keep=False)
# mean_dup = submission0[dup_idx].groupby('row_id')['meter_reading'].mean().reset_index()
# cleaned_submission = pd.concat([submission0.drop_duplicates(subset='row_id', keep=False), mean_dup], axis=0)
# cleaned_submission = cleaned_submission.sort_values('row_id').reset_index(drop=True)
# cleaned_submission.to_csv("submission.csv", index=False, float_format="%.4f")



print('Reading leak data...')
leak_data = pd.read_feather(dataDir/'leak_data_1718.feather',use_threads=True)


print('Mapping row_id to leak data...')
# leak_data2.reset_index().set_index(['timestamp','building_id','meter']).merge(leak_data2.reset_index(),on=['timestamp','building_id','meter'], left_index=True, how='inner')
# leak_row_id = tst_meter.reset_index().merge(leak_data2.reset_index(),on=['timestamp','building_id','meter'],  how='inner')
tst_meter = pd.read_feather(dataDir/'tst_meter.feather', use_threads=True)
leak_row_id = tst_meter.merge(leak_data,on=['timestamp','building_id','meter'],  how='inner')

try:
    del tst_meter
except:
    pass

print('Updating submission...')
cleaned_submission.set_index('row_id', inplace=True)
cleaned_submission = cleaned_submission.squeeze()
cleaned_submission.loc[leak_row_id.row_id] = leak_row_id['meter_reading'].values
cleaned_submission = cleaned_submission.reset_index()
cleaned_submission.sort_values('row_id', inplace=True)
print('Finished.')


print('########################################################################\n'
      'Validate submission\n'
      '########################################################################')

print(f'Submission max: {cleaned_submission.meter_reading.max()}, min: {cleaned_submission.meter_reading.min()}')

print('Checking duplicate...')
check_assertion(not cleaned_submission.row_id.duplicated().any())


print('Checking NA...')
check_assertion(cleaned_submission.notna().all().all())
print('Passed.')

"""  read sample submission  """
print('Checking format with sample_submission')
sample_submission = pd.read_feather(dataDir/'sample_submission.feather')
check_assertion(cleaned_submission.shape == sample_submission.shape)
check_assertion(cleaned_submission.row_id.equals(sample_submission.row_id))
print('Passed.')


print('########################################################################')
print('Writing submission...')
cleaned_submission.to_csv(dataDir / "submission.csv", index=False, float_format="%.4f")

print('All finished.')

print("--- %s seconds ---" % (time.time() - start_time))

print('Press any key to exit...')
input()
raise ValueError('To stop running')


set(sample_submission.row_id).difference(set(cleaned_submission.row_id))

