import numpy as np
import pandas as pd
import os
os.chown()
np.random.seed(12)

def get_lst_range(data):
    iszero = np.concatenate(([0],
                             np.greater(data, 0).view(np.int8),
                             [0]))
    absdiff = np.abs(np.diff(iszero))
    lst_range = np.where(absdiff == 1)[0].reshape(-1, 2)
    return lst_range


def individual_return(data):
    shifted_data_zero = np.concatenate((data[1:], [0]))
    zero_range = np.divide(shifted_data_zero, data)
    zero_range = np.concatenate(([data.iloc[0]], zero_range)) - 1
    zero_range = zero_range[:-1]
    return zero_range


def get_lst_return(df, range_df):
    len_range_df = range_df.diff(axis=1).stack()[::2]
    lst_mask = len_range_df == np.max(len_range_df, axis=0)
    lst_df = range_df[lst_mask]

    result_ls = []
    for fdname in lst_mask.columns:
        single_df = pd.DataFrame(columns=['START', 'END', 'LEN', 'RETURN'])
        for i in lst_df[fdname].dropna().index:
            start = lst_df[fdname].iloc[i]['START'].astype(int)
            pre_start = start - 1 if start != 0 else None
            end = lst_df[fdname].iloc[i]['END'].astype(int)
            real_end = end - 1
            # print(type(start))
            # print(df[fdname].iloc[start-1])
            if pre_start:
                value = df[fdname].iloc[real_end] / df[fdname].iloc[pre_start] - 1
            else:
                value = df[fdname].iloc[real_end] - 1
            single_ser = pd.Series({'START': df.index[start],
                                    'END': df.index[real_end],
                                    'LEN': end - start,
                                    'RETURN': value})
            single_df = single_df.append(single_ser, ignore_index=True)
            single_df = single_df[single_df['RETURN'] == single_df['RETURN'].max()]
        single_df.columns = [[fdname, fdname, fdname, fdname],
                             ['START', 'END', 'LEN', 'RETURN']]
        result_ls.append(single_df)
    result_df = pd.concat(result_ls, axis=1)
    if result_df.isnull().any().any():
        result_df = result_df.fillna(method='ffill').dropna(how='any', axis=0)
    return result_df


df = pd.DataFrame(np.random.uniform(0., 2.5, (18, 5)), columns=list('ABCDE'))
df.index = [20140531, 20140630, 20140731, 20140831, 20140930, 20141031, 20141130, 20141231, 20150131, 20150228,
            20150331, 20150430, 20150531, 20150630, 20150731, 20150831, 20150930, 20151030]

cumprod_df = df.cumprod()

start_date = 20150507

indi_return = cumprod_df.apply(individual_return, axis=0)

freq_prev_idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5])

actual_start = df.index[df.index.searchsorted(start_date, side='right') - 1]

ranges = pd.Series(range(int(start_date <= actual_start), freq_prev_idx.max() + 2)).apply(
    lambda x: (x, freq_prev_idx.searchsorted(x, side='left') - 1))

result_ls = []

for offset in range(len(ranges)):
    r = ranges[offset]
    range_lst_tmp = []
    # fdname = tst_df.columns[fdidx]
    for fdidx in range(df.columns.size):
        fdname = df.columns[fdidx]
        _data = indi_return.iloc[r[0]:r[1] + 1, fdidx]
        single_rst = get_lst_range(_data) + offset
        range_lst_tmp.append(pd.DataFrame(single_rst, columns=[[fdname, fdname], ['START', 'END']]))
    range_df = pd.concat(range_lst_tmp, axis=1)
    range_df.columns.names = ['FUND', 'RANGE']
    range_df.index.names = ['idx']
    result_single_df = get_lst_return(cumprod_df, range_df)
    result_ls.append(result_single_df)
result_df = pd.concat(result_ls, axis=0)
newindex = df.index[df.index.searchsorted(start_date, side='right') - 1:]
result_df.index = newindex
print(result_df.shape)
