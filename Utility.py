import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn import preprocessing
from meteocalc import Temp, dew_point, heat_index, wind_chill, feels_like
import warnings
import lightgbm as lgb

ids = pd.IndexSlice # for multi-indexing

rootDir = Path(r'C:\Users\HAOYANG\Dropbox\Kaggle\ASHRAE')
dataDir = Path(r'C:\Users\HAOYANG\Documents\Kaggle\ASHRAE')
inputDir = dataDir/'train_test_input'
base_data_dir = dataDir/'base_data'
processed_data_dir = dataDir / 'processed_data'
train_output_dir = dataDir/'train_output'
test_output_dir = dataDir/'test_output'


basic_feature_set = ['site_id','building_id','primary_use','square_feet','year_built','floor_count',
               'air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed',
                        'hour','dayofweek']

cat_fts = ('building_id','site_id',  # shared
           'meter',  # train.csv
           'primary_use',  # building_meta.csv
           )
int_fts = ('square_feet', 'year_built', 'floor_count'  # building_meta.csv
           'cloud_coverage', 'precip_depth_1_hr', 'wind_direction'  #weather_train.csv
           )

dtypes_dict = {**{ft: 'float' for ft in int_fts},
               **{ft: 'category' for ft in cat_fts},
               }


def check_assertion(condition):
    if condition:
        print('Passed.')
    else:
        warnings.warn('Failed.')



# def add_time_ft(x, everything):
#     new_ft_cat = ['hour', 'dayofweek', 'month']
#     x['hour'] = everything.index.hour
#     x['dayofweek'] = everything.index.dayofweek
#     x['month'] = everything.index.month
#     x[new_ft_cat] = x[new_ft_cat].astype('category')
#
#
# def add_weather_feature(wthr):
#     complete_tindex = pd.date_range(start=wthr.timestamp.min(), end=wthr.timestamp.max(), freq='1H')
#
#     itpl_wthrs_list = []
#     for site_id, wthr_site in wthr.groupby('site_id'):
#         itpl_wthr_site = wthr_site.set_index('timestamp').reindex(complete_tindex).select_dtypes('number').interpolate(
#             axis=1)
#         diff_lag_3h = (itpl_wthr_site - itpl_wthr_site.shift(3)).add_suffix('_diff_lag_3h')
#         itpl_wthr_site = pd.concat([itpl_wthr_site, diff_lag_3h], axis=1)
#         # todo add check, make sure no other non-numeric column missed
#         # itpl_wthr_site['site_id'] = wthr_site.select_dtypes(exclude='numeric')
#         itpl_wthr_site['site_id'] = site_id
#         itpl_wthrs_list.append(itpl_wthr_site)
#
#     itpl_wthrs = pd.concat(itpl_wthrs_list, axis=0)
#     itpl_wthrs.index.name = 'timestamp'
#     return itpl_wthrs


def convert_UTC_to_LocalTime(wthr):
    assert 'timestamp' in wthr.columns
    """  Calibrate Weather Time Zones  """
    TimeZone_dict = {'US/Eastern': [0, 8, 15, 7, 11, 14, 3, 6],
                     'US/Central': [13, 9],
                     'US/Mountain': [2],
                     'US/Pacific': [4, 10],
                     'Europe/London': [12, 1, 5]}

    TimeZone_dict = {k: list(map(str, v)) for k, v in TimeZone_dict.items()}
    wthr = wthr.copy()
    wthr.rename(columns={'timestamp': 'GMT_timestamp'}, inplace=True)
    wthr.GMT_timestamp = wthr.GMT_timestamp.dt.tz_localize('GMT')
    local_ts = pd.Series(index=wthr.index)
    for tz_name, site_ids in TimeZone_dict.items():
        row_indexer = np.flatnonzero(wthr.site_id.isin(site_ids))
        local_ts[row_indexer] = wthr.loc[row_indexer, 'GMT_timestamp'].dt.tz_convert(tz_name).dt.tz_localize(None)

    wthr['timestamp'] = pd.to_datetime(local_ts)
    wthr.drop(columns='GMT_timestamp', inplace=True)
    return wthr


# def basic_CV(X, y, cv_key, model_constructor, cv_iterator, cv_errs, cv_info,model_kwargs=None,fit_kwargs=None):
#     from sklearn.metrics import mean_squared_error
#     errors = []
#     models = []
#     cv_square_error = np.empty_like(y)
#     cv_square_error[:] = np.nan
#     for train_idx, test_idx in cv_iterator:
#         train_x = X.iloc[train_idx]
#         train_y = y.iloc[train_idx]
#         test_x = X.iloc[test_idx]
#         test_y = y.iloc[test_idx]
#
#         model = model_constructor(**model_kwargs)
#         model.fit(train_x, train_y, **fit_kwargs)
#         test_pred = model.predict(test_x)
#
#         models.append(model)
#         errors.append(np.sqrt(mean_squared_error(y_true=test_y, y_pred=test_pred)))
#         cv_square_error[test_idx] = np.square(test_y - test_pred)
#
#     cv_info[cv_key+'_error'] = cv_square_error
#     cv_errs[cv_key] = errors
#     return models


def basic_CV(X, y, cv_key, model_constructor, cv_iterator, cv_errs, cv_info,model_kwargs=None,fit_kwargs=None):
    from sklearn.metrics import mean_squared_error
    errors = []
    models = []
    cv_square_error = pd.Series(index=y.index)
    for train_idx, test_idx in cv_iterator:
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        test_x = X.loc[test_idx]
        test_y = y.loc[test_idx]

        model = model_constructor(**model_kwargs)
        if fit_kwargs is None:
            model.fit(train_x, train_y,)
        else:
            model.fit(train_x, train_y, **fit_kwargs)
        test_pred = model.predict(test_x)

        models.append(model)
        errors.append(np.sqrt(mean_squared_error(y_true=test_y, y_pred=test_pred)))
        # cv_square_error[test_idx] = np.square(test_y - test_pred)

    # cv_info[cv_key+'_error'] = cv_square_error
    # cv_errs[cv_key] = errors
    return models




def basic_CV2(X, y, cv_key, model_constructor, cv_iterator, cv_errs, cv_info,model_kwargs=None,fit_kwargs=None):

    from sklearn.metrics import mean_squared_error
    errors = []
    models = []
    cv_square_error = pd.Series(index=y.index)
    for train_idx, test_idx in cv_iterator:
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        test_x = X.loc[test_idx]
        test_y = y.loc[test_idx]

        model = model_constructor(**model_kwargs)
        if fit_kwargs is None:
            model.fit(train_x, train_y,)
        else:
            model.fit(train_x, train_y, **fit_kwargs)
        test_pred = model.predict(test_x)


    return models









def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            # Print current column type
            # print("******************************")
            # print("Column: ", col)
            # print("dtype before: ", df[col].dtype)
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # print("min for this col: ", mn)
            # print("max for this col: ", mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
                # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ", df[col].dtype)
            # print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    print()
    return df, NAlist


def reset_catIndex(df):
    # df.select_dtypes('category').apply(lambda x: x.cat.remove_unused_categories(inplace=True))
    for cat_col in df.select_dtypes('category'):
        df[cat_col].cat.remove_unused_categories(inplace=True)


"""  ######################  functions for DATA CLEANING  ##############################  """


def interp(meter_mat, max_gap):
    """
    interpolate NA gaps no larger than max_gap, without the "spilling" effect from the original pandas interpolate function
    :param meter_mat: index: const frequency time index, col: building ids
    :param max_gap: defines the maximum NA gap to be interpolated
    :return: a new matrix
    """
    meter_mat_intrp = meter_mat.interpolate(method='time', limit=max_gap, limit_direction='forward', limit_area='inside')
    mask = meter_mat.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in meter_mat:
        mask[i] = (grp.groupby(i)['ones'].transform('count') <= max_gap) | meter_mat[i].notnull()

    result = meter_mat_intrp.where(mask, np.nan)
    return result


def col_to_mat(mt_gps, meter_type):
    meter = mt_gps.get_group(meter_type).drop(columns='meter')
    meter_mat = meter.set_index('building_id', append=True).unstack()
    meter_mat = meter_mat.droplevel(axis=1, level=0)

    return meter_mat
    # return meter_mat.unstack().rename(f'meter_{meter_type}')


def clean_meter_reading(mt, meter_type=None, max_tol=8, exclude_bid=None, drop_zero=True):
    # meter = mt_gps.get_group(meter_type).drop(columns='meter')
    # meter_mat = meter.set_index('building_id', append=True).unstack()
    # meter_mat = meter_mat.droplevel(axis=1, level=0)
    if isinstance(mt,pd.core.groupby.generic.DataFrameGroupBy):
        meter = mt.get_group(meter_type).drop(columns='meter')
    elif isinstance(mt, pd.DataFrame):
        meter = mt.drop(columns='meter')
    meter_mat = meter.set_index('building_id', append=True).unstack()
    meter_mat = meter_mat.droplevel(axis=1, level=0)
    if exclude_bid is not None:
        set_aside = meter_mat.loc[:, exclude_bid]
        meter_mat_copy = meter_mat.drop(columns=exclude_bid).copy()
    else:
        meter_mat_copy = meter_mat.copy()

    """######################  replace all 0 values with NA  ##############################"""
    if drop_zero:
        meter_mat_copy.where(meter_mat_copy != 0, np.nan, inplace=True)

    """######################  interpolate eligible NA gaps  ##############################"""

    meter_mat_copy = interp(meter_mat_copy, max_gap=6)

    """  ###################  replace eligible consecutive constants segments with NA  ##############################"""

    """
    Construct a matrix indicates locations of consecutive constants
    Falses: consecutive constant segments (CCS)
    Truess: Not consecutive constant (NCC)
    """
    const_locs = (meter_mat_copy.diff() != 0) & (meter_mat_copy.diff(-1) != 0)

    """
    Detect consecutive constants segments with a length no greater than a defined window (max_tol),
    which shall be exempted from removal. 
    And create a mask for such segments
    Trues: eligible segments
    Falses: Non-eligible segments or NCC
    """

    # max_tol = 8

    mask = np.empty_like(const_locs)
    mask[:] = False
    for col_i, (col_name, col_val) in enumerate(const_locs.items()):
        col = col_val.values
        mask_inds = []
        temp_idx_containter = []
        seen_false = False
        for i, x in enumerate(col):
            if x:
                if seen_false is True:
                    seen_false = False
                    if len(temp_idx_containter) <= max_tol:
                        mask_inds.extend(temp_idx_containter)
                    temp_idx_containter = []
            else:
                if seen_false is False:
                    seen_false = True
                temp_idx_containter.append(i)

        # 边界条件
        if len(temp_idx_containter) > 0 and len(temp_idx_containter) <= max_tol:
            mask_inds.extend(temp_idx_containter)
        mask[mask_inds, col_i] = True

    # wrap the mask into a dataframe
    ignor_const_mask = pd.DataFrame(mask, index=const_locs.index, columns=const_locs.columns)

    """
    peform an OR operation to obtain the final mask fo CCS removal
    Falses: CCS
    """
    const_mask = ignor_const_mask | const_locs

    """
    remove eligible CCS
    """
    meter_mat_copy.where(const_mask, np.nan, inplace=True)
    meter_mat_copy.dropna(axis=1, how='all', inplace=True)  # remove all-zero columns

    if exclude_bid is not None:
        meter_mat_copy = pd.concat(([meter_mat_copy,set_aside]),axis=1)
    return meter_mat_copy
    # return meter_mat_copy.unstack().sort_index().rename(f'meter_{meter_type}')


def get_meter_mat_copy(mt):
    assert mt.index.name == 'timestamp'
    meter = mt[['building_id', 'meter_reading']]
    meter_mat = meter.set_index('building_id', append=True).unstack()
    meter_mat = meter_mat.droplevel(axis=1, level=0)
    return meter_mat.copy()


def clean_meter_reading2(mt, max_tol=8, drop_zero=True):
    # if isinstance(mt,pd.core.groupby.generic.DataFrameGroupBy):
    #     meter = mt.get_group(meter_type)['meter_reading']
    # elif isinstance(mt, pd.DataFrame):
    #     pass
    print('Cleaning meter readings...')
    meter_mat_copy = get_meter_mat_copy(mt)
    """######################  replace all 0 values with NA  ##############################"""
    print(f'drop_zero={drop_zero}')
    if drop_zero:
        meter_mat_copy.where(meter_mat_copy != 0, np.nan, inplace=True)
    else:
        zero_mask = meter_mat_copy == 0

    """######################  interpolate eligible NA gaps  ##############################"""
    meter_mat_copy = interp(meter_mat_copy, max_gap=6)

    """  ###################  replace eligible consecutive constants segments with NA  ##############################"""
    print(4)
    print(f'Total: {meter_mat_copy.shape[1]} cols ')
    mask_for_removal = pd.DataFrame(False, index=meter_mat_copy.index, columns=meter_mat_copy.columns)
    for i,(bid, col) in enumerate(meter_mat_copy.iteritems()):
        print(f'Processing col {i}')
        seen = None
        const_count = 0
        to_remove_idx = []
        temp_idx_container = []
        for tstamp, x in col.dropna().items():
            if seen is None:
                seen = x
            elif x == seen:
                const_count += 1
                temp_idx_container.append(tstamp)
            else:
                # print(f'{seen}: {const_count}')
                if const_count > max_tol:
                    to_remove_idx.extend(temp_idx_container)
                temp_idx_container = []
                const_count = 0
                seen = x

        mask_for_removal.loc[to_remove_idx, bid] = True

    """
    remove eligible CCS
    """
    meter_mat_copy.mask(mask_for_removal, np.nan, inplace=True)
    if not drop_zero:
        print('Putting 0s back...')
        meter_mat_copy.mask(zero_mask, other=0, inplace=True)
    meter_mat_copy.dropna(axis=1, how='all', inplace=True)  # remove all-NA columns

    return meter_mat_copy, mask_for_removal


def clean_temporal_bands(mt, bmeta, thresh=0.1, site_thresh_dict=None, visualize=False):
    meter_mat = mt.copy()
    if meter_mat.columns.name == 'building_id' and meter_mat.index.name == 'timestamp':
        meter_mat = meter_mat.T.merge(bmeta[['building_id', 'site_id']], on='building_id', how='left')
        meter_mat = meter_mat.set_index(['site_id', 'building_id']).T
    elif meter_mat.index.name == 'timestamp' and set(['building_id', 'site_id']).issubset(meter_mat.columns):
        meter_mat = meter_mat.set_index(['site_id', 'building_id'], append=True)['meter_reading']
        meter_mat = meter_mat.unstack(['site_id', 'building_id'])

    meter_mat_sgp = meter_mat.groupby(level='site_id', axis=1)
    uni_ratios = meter_mat_sgp.apply(lambda gp: gp.nunique(axis=1) / gp.count(axis=1))

    if site_thresh_dict is not None:
        site_rows_to_remove_list = []
        for site_id, site_thresh in site_thresh_dict.items():
            site_rows_to_remove_list.append(uni_ratios.loc[:, site_id] < site_thresh)

        other_sites_rows_to_remove = uni_ratios.loc[:, uni_ratios.columns.difference(site_thresh_dict.keys())] < thresh
        rows_to_remove = pd.concat([pd.concat(site_rows_to_remove_list, axis=1), other_sites_rows_to_remove], axis=1).loc[:,
        uni_ratios.columns]
    else:
        rows_to_remove = uni_ratios < thresh

    mask_for_removal = meter_mat.T.reset_index()[['site_id', 'building_id']].merge(rows_to_remove.T, on='site_id', how='left').set_index(['site_id', 'building_id']).T


    meter_mat.mask(mask_for_removal, other=np.nan, inplace=True)
    meter_mat.index.name = 'timestamp'
    meter_mat.index = pd.to_datetime(meter_mat.index)

    if visualize:
        """ visualizations """
        plt.figure()
        # sns.heatmap(rows_to_drop.astype(int))
        sns.heatmap(mask_for_removal.astype(int))
        # sns.heatmap(meter_mat)

        uni_ratios.plot()

        fig, axes = plt.subplots(len(meter_mat_sgp))
        for i, (gp_id, gp) in enumerate(meter_mat_sgp):
            (gp.nunique(axis=1) / gp.count(axis=1)).plot(ax=axes[i])
            axes[i].set_ylabel(f'site: {gp_id}')
            axes[i].set_ylim(0, 1)
            axes[i].hlines(thresh, *axes[i].get_xlim(), 'r')

        fig, axes = plt.subplots(1, len(meter_mat_sgp))
        for i, (gp_id, gp) in enumerate(meter_mat_sgp):
            axes[i].imshow(gp, aspect='auto')
            axes[i].set_title(f'site: {gp_id}')

    return meter_mat, mask_for_removal


def clean_isolated_bands(meter_mat, thresh=0.3, exempt_sites=[]):
    meter_mat_sgp = meter_mat.groupby(level='site_id', axis=1)
    nonNA_ratio = meter_mat_sgp.count() / meter_mat_sgp.size()
    nonNA_mask = nonNA_ratio > thresh
    nonNA_mask.isna().any()
    isolated_bands = nonNA_mask.rolling(3, center=True).apply(lambda x: list(x) == [False, True, False], raw=True)
    if exempt_sites:
        print(f'Exempting sites: {exempt_sites}')
        isolated_bands[exempt_sites] = 0

    mask_for_removal = meter_mat.T.reset_index()[['site_id', 'building_id']].merge(isolated_bands.T, on='site_id',
                                                                                   how='left').set_index(
        ['site_id', 'building_id']).T
    mask_for_removal = mask_for_removal == 1
    meter_mat.mask(mask_for_removal, other=np.nan, inplace=True)

    return meter_mat, mask_for_removal


def reformat_meter_mat(meter_mat, bmeta, meter_type):
    if meter_mat.index.name == 'timestamp' and meter_mat.columns.name == 'building_id':
        meter_mat = meter_mat.unstack().dropna().rename('meter_reading').reset_index()
        meter_mat['meter'] = meter_type
        reformat_meter = meter_mat.merge(bmeta[['site_id','building_id']], on='building_id', how='left')
    elif meter_mat.index.name == 'timestamp' and list(meter_mat.columns.names) == ['site_id', 'building_id']:
        reformat_meter = meter_mat.stack(['site_id', 'building_id']).rename('meter_reading').reset_index()
        reformat_meter['meter'] = meter_type

    reformat_meter = reformat_meter.loc[:, ['timestamp', 'building_id', 'meter_reading', 'meter', 'site_id']]
    return reformat_meter


def bd_mat_reformat(bid_sid_indexer, meter_mat, meter_type, sort=True, add_site_id=True,
                    to_series=True, dropna=True):
    meter_mat_T = meter_mat.T
    if sort:
        meter_mat_T.sort_index(inplace=True)
    if add_site_id:
        meter_mat_T['site_id'] = bid_sid_indexer.reindex(meter_mat_T.index)
        meter_mat_T.set_index('site_id', append=True, inplace=True)
        meter_mat_T = meter_mat_T.swaplevel()
        meter_mat_T.columns = pd.to_datetime(meter_mat_T.columns)
    if to_series:
        meter_mat_T = meter_mat_T.T.unstack().rename(f'meter_{meter_type}')
        if dropna:
            meter_mat_T.dropna(inplace=True)
    return meter_mat_T


def intrp_weather(ft_col,ft, max_gap):
    mat_form = ft_col.unstack()
    mat_form_intrp = interp(mat_form, max_gap=max_gap)
    return mat_form_intrp.unstack().rename(ft)



def preprocess_weather_data(wthr_GMT, wthr_max_gap = 6):
    wthr_GMT = wthr_GMT.set_index(['timestamp', 'site_id'])


    """
    interpolate air_temperature, dew_temperature, sea_level_pressure, wind_speed
    keep an eye: cloud_coverage, precip_depth_1_hr
    """
    wthr_to_intrp = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed',
                     'cloud_coverage', 'precip_depth_1_hr']

    other_features = wthr_GMT.drop(wthr_to_intrp, axis=1)
    wthr_intrp1 = pd.concat([intrp_weather(wthr_GMT[ft], ft, wthr_max_gap) for ft in wthr_to_intrp], axis=1)


    wind_dir = wthr_GMT['wind_direction']
    wthr_intrp2 = pd.concat([intrp_weather(np.sin(2 * np.pi * wind_dir / 360), 'wind_dir_sin', wthr_max_gap),
                             intrp_weather(np.cos(2 * np.pi * wind_dir / 360), 'wind_dir_cos', wthr_max_gap)], axis=1)

    """
    Combine all processed features
    """
    wthr_cleaned = pd.concat([wthr_intrp1, wthr_intrp2], axis=1)
    wthr_cleaned = wthr_cleaned.merge(other_features, on=['site_id', 'timestamp'], how='left')

    # return wthr_cleaned
    # wthr_cleaned = convert_UTC_to_LocalTime(wthr_cleaned.reset_index(level=['timestamp', 'site_id']))
    return wthr_cleaned


def cyclic_encoder(x, period, decimal=4):
    x_sin = np.sin(2 * np.pi * x / period)
    x_cos = np.cos(2 * np.pi * x / period)
    return np.around(np.stack((x_sin,x_cos), axis=1), decimals=decimal)


# def append_time_feature(meter,):
#     print('Appending time feature...')
#     timestamp = meter.index.get_level_values('timestamp')
#     meter['hour'] = timestamp.hour
#     meter['dayofweek'] = timestamp.dayofweek
#     meter['month'] = timestamp.month
#     # meter['dayofmonth'] = timestamp.days_in_month
#     # return meter


def append_time_feature2(meter,):
    print('Appending time feature...')
    if 'timestamp' in meter:
        from_index = False
    else:
        index_names = meter.index.names
        assert 'timestamp' in index_names
        meter.reset_index(inplace=True)
        from_index = True

    meter['hour'] = meter.timestamp.dt.hour
    meter['dayofweek'] = meter.timestamp.dt.dayofweek
    meter['month'] = meter.timestamp.dt.month
    meter['dayofyear'] = meter.timestamp.dt.dayofyear
    # meter['dayofmonth'] = timestamp.days_in_month
    if from_index:
        meter.set_index(index_names, inplace=True)


def append_bmeta_wthr(meter, bmeta, wthr):
    new_meter = meter.to_frame().join(bmeta, on=['site_id', 'building_id'], how='left')
    new_meter = new_meter.join(wthr, on=['site_id', 'timestamp'], how='left')
    return new_meter



def get_label_encoders(col, lb_encoder_dict):
    col_name = col.name
    le = preprocessing.LabelEncoder()
    col = col.astype(str)  # convert to string, so np.nans don't get messed up by this encoder
    uni_labels = np.sort(col.unique())
    print(uni_labels)
    le.fit(uni_labels)
    col = le.transform(col)
    lb_encoder_dict[col_name] = le
    return col


def auto_lb_transform(df,lb_encoder_dict):
    print('Encoding cat labels...')
    for col_name, col in df.iteritems():
        if col_name in lb_encoder_dict.keys():
            print(f'Label encoding: {col_name}')
            try:
                df[col_name] = lb_encoder_dict[col_name].transform(col.astype(str))
            except:
                warnings.warn(f'Skipped encoding {col_name}.\n Check its contents.')


def add_time_lag_diff_feature(wthr_cleaned, lag_interval=6, total_span=48):
    wthr_gps = wthr_cleaned[['air_temperature', 'dew_temperature', 'sea_level_pressure', 'radiation', 'wind_speed', 'cloud_coverage']].groupby('site_id')
    t_lags = np.arange(lag_interval,total_span+lag_interval,lag_interval)
    tlag_wthr_fts_sites = {}
    for site_id, wthr_one_site in wthr_gps:
        tlag_wthr_fts = {}
        for tlag in t_lags:
            tlag_wthr_fts[tlag] = (wthr_one_site - wthr_one_site.shift(tlag)).add_suffix(f'_diff_lag_{tlag}')
            tlag_wthr_fts[-tlag] = (wthr_one_site - wthr_one_site.shift(-tlag)).add_suffix(f'_diff_lag_{-tlag}')

        tlag_wthr_fts_sites[site_id] = pd.concat(tlag_wthr_fts.values(), axis=1)

    tlag_wthr = pd.concat(tlag_wthr_fts_sites.values(), axis=0)
    return pd.concat([wthr_cleaned, tlag_wthr], axis=1)


def add_time_lag_diff_feature2(wthr_cleaned, features, lag_interval=6, total_span=48):
    wthr_gps = wthr_cleaned[features].groupby('site_id')
    t_lags = np.arange(lag_interval,total_span+lag_interval,lag_interval)
    tlag_wthr_fts_sites = {}
    for site_id, wthr_one_site in wthr_gps:
        tlag_wthr_fts = {}
        for tlag in t_lags:
            tlag_wthr_fts[tlag] = (wthr_one_site - wthr_one_site.shift(tlag)).add_suffix(f'_diff_lag_{tlag}')
            tlag_wthr_fts[-tlag] = (wthr_one_site - wthr_one_site.shift(-tlag)).add_suffix(f'_diff_lag_{-tlag}')

        tlag_wthr_fts_sites[site_id] = pd.concat(tlag_wthr_fts.values(), axis=1)

    tlag_wthr = pd.concat(tlag_wthr_fts_sites.values(), axis=0)
    return tlag_wthr


# https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type
def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby(level='site_id', as_index=False)
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
            'wind_dir_sin', 'wind_dir_cos', 'wind_speed','radiation','azimuth','altitude_deg']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().astype(np.float16).add_suffix(f'_mean_lag{window}').droplevel(level=0)
    lag_max = rolled.max().astype(np.float16).add_suffix(f'_max_lag{window}').droplevel(level=0)
    lag_min = rolled.min().astype(np.float16).add_suffix(f'_min_lag{window}').droplevel(level=0)
    lag_std = rolled.std().astype(np.float16).add_suffix(f'_std_lag{window}').droplevel(level=0)

    return pd.concat([weather_df, lag_mean, lag_max, lag_min, lag_std], axis=1)

# plt.close('all')


"""  4-4-4 month train-skip-valid scheme  """

def get_cyc_list_splits(start, seg_len, elements = list(range(1,13))):
    from itertools import cycle
    cyc_list = cycle(elements)

    element_segments = []
    i = 1
    while i < start:
        next(cyc_list)
        i += 1

    total_len = 0
    while total_len < len(elements):
        segment = []
        for j in range(seg_len):
            segment.append(next(cyc_list))
            total_len += 1
        element_segments.append(segment)

    return element_segments



def get_month_folds(start, seg_len, elements=list(range(1, 13)), year=16):
    segs = get_cyc_list_splits(start, seg_len, elements=elements)

    folds_str = []
    for seg in segs:
        # folds_str.append(['20{year}-{:02d}'.format(elm) for elm in seg])
        folds_str.append([f'20{year}-{elm:02d}' for elm in seg])
    return folds_str




def lgb_CV3(X, y, cv_iterator, model_params, train_params=None ):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    train_pred_dict = {}
    valid_pred_dict = {}
    test_pred_dict = {}
    evals_result_dict = {}
    models_dict = {}
    cv_scores = {}
    cv_names = []

    for cv_key, (train_idx, valid_idx, test_idx) in cv_iterator.items():
        print("******************************************************")
        print(f'Evaluating: {cv_key}')
        evals_result = {}
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        valid_x = X.loc[valid_idx]
        valid_y = y.loc[valid_idx]
        test_x = X.loc[test_idx]
        test_y = y.loc[test_idx]

        d_train = lgb.Dataset(train_x, label=train_y)
        d_valid = lgb.Dataset(valid_x, label=valid_y)
        # d_test = lgb.Dataset(test_x, label=test_y)
        booster = lgb.train(model_params, d_train,
                            num_boost_round=3000,
                            valid_sets=[d_train, d_valid],
                            valid_names=['train', 'valid', ],
                            early_stopping_rounds=50,
                            evals_result=evals_result,
                            verbose_eval=10)

        models_dict[cv_key] = booster

        train_pred = pd.DataFrame(booster.predict(train_x), index=train_y.index, columns=['pred_reading'])
        valid_pred = pd.DataFrame(booster.predict(valid_x), index=valid_y.index, columns=['pred_reading'])
        test_pred = pd.DataFrame(booster.predict(test_x), index=test_y.index, columns=['pred_reading'])

        train_pred_dict[cv_key] = train_pred
        valid_pred_dict[cv_key] = valid_pred
        test_pred_dict[cv_key] = test_pred
        evals_result_dict[cv_key] = evals_result
        test_score = np.sqrt(mean_squared_error(test_y, test_pred))
        print(f'test\'s rmes: {test_score}')
        cv_scores[cv_key] = dict(train=booster.best_score['train']['rmse'],
                                 valid=booster.best_score['valid']['rmse'],
                                 test=test_score),
        cv_names.append((cv_key))

    return {'train_pred': train_pred_dict, 'valid_pred': valid_pred_dict, 'test_pred': test_pred_dict,
                'eval_results': evals_result_dict, 'models': models_dict, 'scores': cv_scores, 'cv_names' : cv_names}

# TODO get best models; get scores on train, valid, test data

def lgb_CV4(X, y, cv_iterator, model_params, early_stop_rounds = 20,train_params=None,external_data=None):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    def make_pred_df(_x, _y, name):
        return pd.DataFrame(np.stack([booster.predict(_x),_x.site_id.values],axis=1), index=_y.index, columns=[f'{name}_pred','site_id']).astype({'site_id':int}, copy=False)

    cv_results = {}

    for cv_key, (train_idx, valid_idx, test_idx) in cv_iterator.items():
        print("******************************************************")
        print(f'Evaluating: {cv_key}')
        evals_result = {}
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        valid_x = X.loc[valid_idx]
        valid_y = y.loc[valid_idx]
        test_x = X.loc[test_idx]
        test_y = y.loc[test_idx]

        d_train = lgb.Dataset(train_x, label=train_y)
        d_valid = lgb.Dataset(valid_x, label=valid_y)
        # d_test = lgb.Dataset(test_x, label=test_y)
        # d_external =
        booster = lgb.train(model_params, d_train,
                            num_boost_round=1500,
                            valid_sets=[d_train, d_valid],
                            valid_names=['train', 'valid', ],
                            early_stopping_rounds=early_stop_rounds,
                            evals_result=evals_result,
                            verbose_eval=10)

        # train_pred = pd.DataFrame(np.stack([booster.predict(train_x),train_x.site_id.values],axis=1), index=train_y.index, columns=['train_pred','site_id']).astype({'site_id':int}, copy=False)
        train_pred = make_pred_df(train_x,train_y,'train')
        valid_pred = make_pred_df(valid_x,valid_y,'valid')
        test_pred = make_pred_df(test_x,test_y,'test')
        test_score = np.sqrt(mean_squared_error(test_y, test_pred['test_pred']))
        print(f'test\'s rmes: {test_score}')
        cv_scores = {'train': booster.best_score['train']['rmse'],
                     'valid': booster.best_score['valid']['rmse'],
                     'test': test_score}
        cv_results[cv_key] = dict(model=booster, train_pred=train_pred, valid_pred=valid_pred, test_pred=test_pred,
                                  evals_result=evals_result, cv_scores=cv_scores)


    return cv_results



# for xgboost
def lgb_CV5(X, y, cv_iterator, model_params, early_stop_rounds = 20,train_params=None,external_data=None):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    def make_pred_df(dm_x,_x, _y, name):
        return pd.DataFrame(np.stack([booster.predict(dm_x),_x.site_id.values],axis=1), index=_y.index, columns=[f'{name}_pred','site_id']).astype({'site_id':int}, copy=False)

    cv_results = {}

    for cv_key, (train_idx, valid_idx, test_idx) in cv_iterator.items():
        print("******************************************************")
        print(f'Evaluating: {cv_key}')
        evals_result = {}
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        valid_x = X.loc[valid_idx]
        valid_y = y.loc[valid_idx]
        test_x = X.loc[test_idx]
        test_y = y.loc[test_idx]

        d_train = xgb.DMatrix(train_x, label=train_y)
        d_valid = xgb.DMatrix(valid_x, label=valid_y)
        d_test = xgb.DMatrix(test_x, label=test_y)
        # d_external =
        booster = xgb.train(model_params, d_train,
                            num_boost_round=600,
                            evals=[(d_valid,'valid')],
                            early_stopping_rounds=early_stop_rounds,
                            # evals_result=evals_result,
                            # verbose_eval=10
                            )

        # train_pred = pd.DataFrame(np.stack([booster.predict(train_x),train_x.site_id.values],axis=1), index=train_y.index, columns=['train_pred','site_id']).astype({'site_id':int}, copy=False)
        train_pred = make_pred_df(d_train,train_x,train_y,'train')
        valid_pred = make_pred_df(d_valid,valid_x,valid_y,'valid')
        test_pred = make_pred_df(d_test,test_x,test_y,'test')
        test_score = np.sqrt(mean_squared_error(test_y, test_pred['test_pred']))
        print(f'test\'s rmes: {test_score}')
        cv_scores = {
                     'valid': booster.best_score,
                     'test': test_score}
        cv_results[cv_key] = dict(model=booster, train_pred=train_pred, valid_pred=valid_pred, test_pred=test_pred,
                                  evals_result=evals_result, cv_scores=cv_scores)


    return cv_results




def lgb_CV6(X, y, cv_iterator, model_params, early_stop_rounds = 20,train_params=None,external_data=None):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    def make_pred_df(_x, _y, name):
        return pd.DataFrame(np.stack([booster.predict(_x),_x.site_id.values],axis=1), index=_y.index, columns=[f'{name}_pred','site_id']).astype({'site_id':int}, copy=False)

    cv_results = {}

    for cv_key, (train_idx, valid_idx, test_idx) in cv_iterator.items():
        print("******************************************************")
        print(f'Evaluating: {cv_key}')
        evals_result = {}
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        valid_x = X.loc[valid_idx]
        valid_y = y.loc[valid_idx]
        test_x = X.loc[test_idx]
        test_y = y.loc[test_idx]

        d_train = lgb.Dataset(train_x, label=train_y)
        d_valid = lgb.Dataset(valid_x, label=valid_y)
        d_test = lgb.Dataset(test_x, label=test_y)
        # d_external =
        booster = lgb.train(model_params, d_train,
                            num_boost_round=1500,
                            valid_sets=[d_train, d_valid, d_test],
                            valid_names=['train', 'valid', 'test'],
                            early_stopping_rounds=early_stop_rounds,
                            evals_result=evals_result,
                            verbose_eval=10)

        # train_pred = pd.DataFrame(np.stack([booster.predict(train_x),train_x.site_id.values],axis=1), index=train_y.index, columns=['train_pred','site_id']).astype({'site_id':int}, copy=False)
        train_pred = make_pred_df(train_x,train_y,'train')
        valid_pred = make_pred_df(valid_x,valid_y,'valid')
        test_pred = make_pred_df(test_x,test_y,'valid')
        # test_pred = make_pred_df(test_x,test_y,'test')
        # test_score = np.sqrt(mean_squared_error(test_y, test_pred['test_pred']))
        # print(f'test\'s rmes: {test_score}')
        cv_scores = {'train': booster.best_score['train']['rmse'],
                     'valid': booster.best_score['valid']['rmse'],
                     'test': booster.best_score['test']['rmse'],
                     # 'test': test_score
                     }
        cv_results[cv_key] = dict(model=booster, train_pred=train_pred, valid_pred=valid_pred, test_pred=test_pred,
                                  evals_result=evals_result, cv_scores=cv_scores)

    return cv_results


def lgb_train(X, y, cv_iterator, model_params, train_params=None ):
    import lightgbm as lgb

    # train_pred_dict = {}
    # valid_pred_dict = {}
    evals_result_dict = {}
    models_dict = {}
    cv_scores = {}

    for cv_key, (fold_A, fold_B, fold_C) in cv_iterator.items():
        print("******************************************************")
        print(f'Evaluating: {cv_key}')
        train_idx = fold_A + fold_B
        valid_idx = fold_C
        evals_result = {}
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        valid_x = X.loc[valid_idx]
        valid_y = y.loc[valid_idx]

        d_train = lgb.Dataset(train_x, label=train_y)
        d_valid = lgb.Dataset(valid_x, label=valid_y)
        # d_test = lgb.Dataset(test_x, label=test_y)
        booster = lgb.train(model_params, d_train,
                            num_boost_round=10000,
                            valid_sets=[d_train, d_valid],
                            valid_names=['train', 'valid', ],
                            early_stopping_rounds=50,
                            evals_result=evals_result,
                            verbose_eval=10)

        models_dict[cv_key] = booster

        # train_pred = pd.DataFrame(booster.predict(train_x), index=train_y.index, columns=['pred_reading'])
        # valid_pred = pd.DataFrame(booster.predict(valid_x), index=valid_y.index, columns=['pred_reading'])

        # train_pred_dict[cv_key] = train_pred
        # valid_pred_dict[cv_key] = valid_pred
        evals_result_dict[cv_key] = evals_result

        cv_scores[cv_key] = dict(train=booster.best_score['train']['rmse'],
                                 valid=booster.best_score['valid']['rmse'],
                                 ),

    return {#'train_pred': train_pred_dict, 'valid_pred': valid_pred_dict,
                'eval_results': evals_result_dict, 'models': models_dict, 'scores': cv_scores}




def lgb_train2(train_x, train_y, valid_x, valid_y, model_params, train_params=None ):
    import lightgbm as lgb
    evals_result = {}

    d_train = lgb.Dataset(train_x, label=train_y)
    d_valid = lgb.Dataset(valid_x, label=valid_y)
    booster = lgb.train(model_params, d_train,
                        num_boost_round=2000,
                        valid_sets=[d_train, d_valid],
                        valid_names=['train', 'valid', ],
                        early_stopping_rounds=20,
                        evals_result=evals_result,
                        verbose_eval=10)

    train_pred = pd.DataFrame(booster.predict(train_x), index=train_y.index, columns=['pred_reading'])
    valid_pred = pd.DataFrame(booster.predict(valid_x), index=valid_y.index, columns=['pred_reading'])


    cv_scores = dict(train=booster.best_score['train']['rmse'],
                     valid=booster.best_score['valid']['rmse'],
                     ),

    return {'train_pred': train_pred, 'valid_pred': valid_pred,
            'eval_results': evals_result, 'models': booster, 'scores': cv_scores}



def lgb_train3(X, y, external_valid_x, external_valid_y, cv_iterator, model_params, train_params=None ):
    import lightgbm as lgb

    train_pred_dict = {}
    valid_pred_dict = {}
    evals_result_dict = {}
    models_dict = {}
    cv_scores = {}

    for cv_key, (fold_A, fold_B, fold_C) in cv_iterator.items():
        print("******************************************************")
        print(f'Evaluating: {cv_key}')
        train_idx = fold_A + fold_B
        valid_idx = fold_C
        evals_result = {}
        train_x = X.loc[train_idx]
        train_y = y.loc[train_idx]
        valid_x = X.loc[valid_idx]
        valid_y = y.loc[valid_idx]

        d_train = lgb.Dataset(train_x, label=train_y)
        d_valid = lgb.Dataset(valid_x, label=valid_y)
        d_valid2 = lgb.Dataset(external_valid_x, label=external_valid_y)
        booster = lgb.train(model_params, d_train,
                            num_boost_round=2000,
                            valid_sets=[d_train, d_valid, d_valid2],
                            valid_names=['train', 'valid','valid2' ],
                            early_stopping_rounds=20,
                            evals_result=evals_result,
                            verbose_eval=10)

        models_dict[cv_key] = booster

        train_pred = pd.DataFrame(booster.predict(train_x), index=train_y.index, columns=['pred_reading'])
        valid_pred = pd.DataFrame(booster.predict(valid_x), index=valid_y.index, columns=['pred_reading'])

        train_pred_dict[cv_key] = train_pred
        valid_pred_dict[cv_key] = valid_pred
        evals_result_dict[cv_key] = evals_result

        cv_scores[cv_key] = dict(train=booster.best_score['train']['rmse'],
                                 valid=booster.best_score['valid']['rmse'],
                                 ),

    return {'train_pred': train_pred_dict, 'valid_pred': valid_pred_dict,
                'eval_results': evals_result_dict, 'models': models_dict, 'scores': cv_scores}



def get_week_folds(tr_meters,mtype):
    period_index = tr_meters[mtype].index.to_period("W")
    period_index.name = 'period'
    u_pidx = period_index.unique().sort_values().tolist()

    fold1 = []
    fold2 = []
    fold3 = []

    while u_pidx:
        if u_pidx:
            fold1.append(u_pidx.pop())
        if u_pidx:
            fold2.append(u_pidx.pop())
        if u_pidx:
            fold3.append(u_pidx.pop())

    folds = {'cv1':(fold1,fold2,fold3),'cv2': (fold2,fold3,fold1), 'cv3': (fold3,fold1,fold2),}
    return period_index, folds


def get_month_folds(tr_meters, mtype):
    fold1_m = np.array([1, 4, 7, 10])
    fold1 = [f'2016-{m:02d}' for m in fold1_m]
    fold2 = [f'2016-{m:02d}' for m in (fold1_m + 1)]
    fold3 = [f'2016-{m:02d}' for m in (fold1_m + 2)]

    folds = {'cv1': (fold1, fold2, fold3), 'cv2': (fold2, fold3, fold1), 'cv3': (fold3, fold1, fold2), }

    period_index = tr_meters[mtype].index.to_period("M")
    period_index.name = 'period'
    return period_index, folds


def get_month_folds2(X):
    fold1_m = np.array([1, 4, 7, 10])
    fold1 = [f'2016-{m:02d}' for m in fold1_m]
    fold2 = [f'2016-{m:02d}' for m in (fold1_m + 1)]
    fold3 = [f'2016-{m:02d}' for m in (fold1_m + 2)]

    folds = {'cv1': (fold1, fold2, fold3), 'cv2': (fold2, fold3, fold1), 'cv3': (fold3, fold1, fold2), }

    period_index = X.index.to_period("M")
    period_index.name = 'period'
    return period_index, folds


def get_month_folds3(tr_meters, mtype):
    fold1_m = np.array([1, 4, 7, 10])
    fold1 = [f'2016-{m:02d}' for m in fold1_m] + [f'2017-{m:02d}' for m in (fold1_m + 1)] + [f'2018-{m:02d}' for m in
                                                                                             (fold1_m + 2)]
    fold2 = [f'2016-{m:02d}' for m in (fold1_m + 1)] + [f'2017-{m:02d}' for m in (fold1_m + 2)] + [f'2018-{m:02d}' for m
                                                                                                   in fold1_m]
    fold3 = [f'2016-{m:02d}' for m in (fold1_m + 2)] + [f'2017-{m:02d}' for m in fold1_m] + [f'2018-{m:02d}' for m in
                                                                                             (fold1_m + 1)]
    # sorted(map(lambda x: x.split('-')[-1], fold3))
    folds = {'cv1': (fold1, fold2, fold3), 'cv2': (fold2, fold3, fold1), 'cv3': (fold3, fold1, fold2), }

    period_index = tr_meters[mtype].index.to_period("M")
    period_index.name = 'period'
    return period_index, folds


def detect_outlier(trace, window=19,contamination=0.05,score_thresh=1.8,prominence=1,rel_height=0.9):
    trace_no0 = trace.replace({0: np.nan})
    trace_itp = trace_no0.dropna()
    # window = 19
    rolling_step = 1
    expanded = pd.concat([trace_itp]*window, axis=1)
    if expanded.columns.nlevels == 2:
        expanded = expanded.droplevel(level=0, axis=1)
    elif expanded.columns.nlevels == 1:
        pass
    else:
        assert expanded.columns.nlevels in [1,2]  # force an error

    half_win, _ = divmod(window, 2)
    expanded.columns = list(range(-half_win, half_win+1))

    shift_cols = []
    for shift, col in expanded.iteritems():
        shift_cols.append(col.shift(shift))

    rolling_vecs = pd.concat(shift_cols, axis=1)

    rolling_vecs2 = rolling_vecs.copy()
    rolling_vecs2 = rolling_vecs2.fillna(method='ffill', axis=0)
    rolling_vecs2 = rolling_vecs2.fillna(method='bfill', axis=0)
    rolling_vecs3 = rolling_vecs2.resample(f'{rolling_step}h').asfreq().dropna(axis=0)
    assert rolling_vecs2.notna().all().all()

    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=max(50, np.round(window/rolling_step).astype(int)),
                             contamination=contamination, novelty=False, #metric='cosine',
                             n_jobs=12)
    lof.fit(rolling_vecs3,)
    lof_score = -lof.negative_outlier_factor_
    lof_score = pd.Series(lof_score, index=rolling_vecs3.index)

    # if rolling_step > 1:
    #     lof_score = lof_score.resample('1h').interpolate()


    major_thresh = lof_score.quantile(0.95)
    major_score = lof_score[lof_score < major_thresh]
    # score_thresh = major_score.mean() + 4*major_score.std()
    # score_thresh = 1.8

    """  Find peaks in LOF scores and get peak stats  """
    from scipy.signal import find_peaks, peak_widths
    # print(f'prominence={4*major_score.mad()}')
    peaks = find_peaks(lof_score, height=score_thresh, distance=window/2, prominence=prominence, wlen=24*3)

    prominences=(peaks[1]['prominences'],
    peaks[1]['left_bases'],
    peaks[1]['right_bases'])

    pws = peak_widths(lof_score, peaks[0], rel_height=rel_height, prominence_data=prominences)


    peak_starts = np.round(pws[2]).astype(int)
    peak_stops = np.round(pws[3]).astype(int)
    peak_intervals = list(zip(peak_starts,peak_stops))

    lof_peak_int_idx = []
    for start_stop in peak_intervals:
        lof_peak_int_idx.extend(np.arange(start_stop[0],start_stop[1]+1))

    lof_peak_int_idx = sorted(list(set(lof_peak_int_idx)))

    otl_mask = pd.Series(False, index=trace.index)
    otl_mask[lof_score.index[lof_peak_int_idx]] = True

    return otl_mask





"""
Adopted from   https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
"""

def make_is_bad_zero(Xy_subset, min_interval=48, summer_start=3000, summer_end=7500):
    """Helper routine for 'find_bad_zeros'.

    This operates upon a single dataframe produced by 'groupby'. We expect an
    additional column 'meter_id' which is a duplicate of 'meter' because groupby
    eliminates the original one."""
    meter = Xy_subset.meter_id.iloc[0]
    is_zero = Xy_subset.meter_reading == 0
    if meter == 0:
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero

    transitions = (is_zero != is_zero.shift(1))
    all_sequence_id_s = transitions.cumsum()
    id_s = all_sequence_id_s[is_zero].rename("id_s")
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        keep = set(id_s[(Xy_subset['time_delta'] < summer_start) |
                       (Xy_subset['time_delta'] > summer_end)].unique())
        is_bad = id_s.isin(keep) & (id_s.map(id_s.value_counts()) >= min_interval)
    elif meter == 1:
        time_id_s = id_s.to_frame().join(Xy_subset['time_delta']).set_index("time_delta").id_s
        is_bad = id_s.map(id_s.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_id_s.get(0, False)
        dec_id = time_id_s.get(8283, False)
        if (jan_id and dec_id and jan_id == time_id_s.get(500, False) and
                dec_id == time_id_s.get(8783, False)):
            is_bad = is_bad & (~(id_s.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result


def find_bad_zeros(Xy):
    """Returns an Index object containing only the rows which should be deleted."""
    # Xy = X.assign(meter_reading=y, meter_id=X.meter)
    # Xy['meter_id'] = Xy.meter
    Xy = Xy.assign(meter_id=Xy.meter)
    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)
    # return is_bad_zero
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])


def find_bad_building1099(Xy):
    """Returns indices of bad rows (with absurdly high readings) from building 1099."""
    return Xy[(Xy.building_id == 1099) & (Xy.meter == 2) & (Xy['meter_reading'] > np.log1p(3e4))].index



"""
FEATURE ENGINEERING
"""

def calculate_rh(df):
    """relative_humidity"""
    return 100 * (np.exp((17.625 * df['dew_temperature']) / (243.04 + df['dew_temperature'])) / np.exp(
            (17.625 * df['air_temperature']) / (243.04 + df['air_temperature'])))



def creat_features(wthr_bmeta):
    print('Creating features...')
    wthr_bmeta['radiation_area'] = wthr_bmeta['radiation'].multiply(wthr_bmeta['square_feet'])
    wthr_bmeta['radiation_cloud_area'] = wthr_bmeta['radiation_cloud'].multiply(wthr_bmeta['square_feet'])
    wthr_bmeta['radiation_cloud_floor'] = wthr_bmeta['radiation_cloud'].multiply(wthr_bmeta['floor_count'])
    wthr_bmeta['wind_area'] = wthr_bmeta['wind_speed'].multiply(wthr_bmeta['square_feet'])
    wthr_bmeta['wind_floor'] = wthr_bmeta['wind_speed'].multiply(wthr_bmeta['floor_count'])
    # wthr_bmeta.set_index('timestamp', inplace=True)
    """
    append time feature
    """
    # append_time_feature(wthr_bmeta)
    # wthr_bmeta = append_time_feature(wthr_bmeta)
    # return wthr_bmeta

def get_holidays(df):
    print('Appending holidays...')
    # df = comb_all_meters.sample(10000).copy()
    assert df.index.name == 'timestamp'
    holidays = pd.read_excel(rootDir / 'holidays_data.xlsx', parse_dates=['date'])
    site_info = pd.read_excel('external_site_info.xlsx')

    holidays_site = holidays.merge(site_info[['site_id', 'country']], on='country')
    holidays_site.date = holidays_site.date.dt.to_period('D')

    df['date'] = df.index.to_period('D')
    to_append = \
    df[['date', 'site_id']].merge(holidays_site.drop(columns='country'), on=['date', 'site_id'], how='left')[
        'is_holiday']
    df['is_holiday'] = to_append.values
    df['is_holiday'].fillna(0, inplace=True)
    df['is_holiday'] = df['is_holiday'].astype(int, copy=False)
    df.drop(columns='date', inplace=True)


def get_holidays2(df):
    print('Appending holidays2...')
    assert df.index.name == 'timestamp'
    holidays = pd.read_excel(rootDir / 'holidays_data.xlsx', parse_dates=['date'])
    site_info = pd.read_excel('external_site_info.xlsx')

    tstamp = pd.period_range(start='2015-12-24', end='2019-1-10', freq='D').to_timestamp()
    holidays_new_all_countries = []
    for country, country_holiday in holidays.groupby('country'):
        is_holiday_new = pd.DataFrame(tstamp.dayofweek.isin([5, 6]).astype(int), index=tstamp,
                                      columns=['is_holiday(new)'])
        is_holiday_new.index.name = 'date'
        is_holiday_new.loc[pd.to_datetime(['2015-12-25', '2015-12-26', '2019-1-1'])] = 1
        is_holiday_new.loc[country_holiday.date] = 1
        is_holiday_new['country'] = country
        is_holiday_new['holiday_density'] = smooth_gaussian(is_holiday_new['is_holiday(new)'], window=15, std=3)
        holidays_new_all_countries.append(is_holiday_new)

    holidays_new_all_countries = pd.concat(holidays_new_all_countries, axis=0)

    holidays_new_all_sites = holidays_new_all_countries.reset_index().merge(site_info[['site_id', 'country']],
                                                                            on='country')
    holidays_new_all_sites.date = holidays_new_all_sites.date.dt.to_period('D')

    df['date'] = df.index.to_period('D')

    return df.reset_index().merge(holidays_new_all_sites[['date', 'is_holiday(new)', 'holiday_density', 'site_id']],
                    on=['date', 'site_id'], how='left').drop(columns='date').set_index('timestamp').astype({'is_holiday(new)':int}, copy=False)



def get_lag_pca(wthr, features_for_lag, lag_interval, total_span, n_pcs=3):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    lag_fts = add_time_lag_diff_feature2(wthr, features_for_lag, lag_interval=lag_interval, total_span=total_span).sort_index()

    pca = PCA(n_components=n_pcs, whiten=True)
    # lag_data = lag_site.filter(regex='air_temperature_diff_lag_.*').dropna().T.sort_index().T
    pca_lag_list = []
    for ft in features_for_lag:
        lag_data = lag_fts.filter(regex=f'{ft}_diff_lag_.*').dropna()
        pca_lag = pca.fit_transform(preprocessing.scale(lag_data))
        pca_lag_df = pd.DataFrame(pca_lag, index=lag_data.index,
                                  columns=[f'{ft}_span{total_span}_pc{i}' for i in range(1, n_pcs + 1)])
        pca_lag_list.append(pca_lag_df)

    return pd.concat(pca_lag_list, axis=1)


def smooth_gaussian(data,window,std):
    from scipy import signal
    g = signal.gaussian(window,std,sym=True)
    con = np.convolve(g/g.sum(),data,mode='valid')
    con_shift = np.r_[np.full(int(window*0.5),np.nan),con,np.full(int(window*0.5),np.nan)]
    return con_shift

"""
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

"""

# plt.close('all')



























































