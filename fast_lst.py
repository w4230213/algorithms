import numpy as np
import pandas as pd

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


def get_answer2(cumprod_df, indi_return, ranges):
    result_d = {}
    cumprod_arr = cumprod_df.values
    indi_arr = indi_return.values
    for offset in range(len(ranges)):
        r = ranges[offset]
        observe_point = indi_return.index[r[1]]
        # print(observe_point)
        single_range_d = {}
        for fdidx in range(cumprod_df.columns.size):
            fdname = cumprod_df.columns[fdidx]
            _data = indi_arr.T[fdidx][r[0]:r[1] + 1]
            single_rst = get_lst_range(_data) + offset
            longest_range = single_rst[np.where(
                np.diff(single_rst) == np.max(np.diff(single_rst)))[0]]
            #         print('-----------')
            #         print('offset:',offset)
            #         print('{}:{}->'.format(observe_point, fdname))
            #         print(longest_range)
            max_return = (0, None, None)
            if longest_range.shape[0] > 1:
                for i in longest_range:
                    pre_start = i[0] - 1 if i[0] != 0 else None
                    real_end = i[1] - 1
                    if pre_start is not None:
                        value = cumprod_arr.T[fdidx][real_end] / \
                                cumprod_arr.T[fdidx][pre_start] - 1
                        # print('case1:',value)
                    else:
                        value = cumprod_arr.T[fdidx][real_end] - 1
                        # print('case2:',value)
                    max_return = (
                        value, i[0], i[1]) if value > max_return[0] else max_return
            else:
                pre_start = longest_range[0][0] - \
                            1 if longest_range[0][0] != 0 else None
                real_end = longest_range[0][1] - 1
                if pre_start is not None:
                    value = cumprod_arr.T[fdidx][real_end] / \
                            cumprod_arr.T[fdidx][pre_start] - 1
                    # print('case3:',value)
                else:
                    value = cumprod_arr.T[fdidx][real_end] - 1
                    # print('case4:',value)
                # print(value)
                max_return = (
                    value, longest_range[0][0], longest_range[0][1]) if value > max_return[0] else max_return
            # print('max_return:',max_return)
            col_return, col_start, col_end = '{}_return'.format(
                fdname), '{}_start'.format(fdname), '{}_end'.format(fdname),
            single_d = {col_return: max_return[0], col_start: max_return[1], col_end: max_return[2]}
            single_range_d.update(single_d)
        result_d[observe_point] = single_range_d
    return result_d


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

result_d = get_answer2(cumprod_df, indi_return, ranges)
mid_result_df = pd.DataFrame(result_d).T
