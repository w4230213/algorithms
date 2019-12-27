import copy
import array
import numpy as np


def merge_sort(a):
    n = len(a)
    aux = np.zeros(shape=(n,))

    def merge(a, lo, mid, hi):
        #n = len(a)
        #aux = np.zeros(shape=(n,))
        for i in range(lo,hi+1):
            aux[i] = a[i]
        i = lo
        j = mid + 1
        for k in range(lo, hi+1):
            if i > mid:
                a[k] = aux[j]
                j += 1
            elif j > hi:
                a[k] = aux[i]
                i += 1
            elif aux[j] < aux[i]:
                a[k] = aux[j]
                j += 1
            else:
                a[k] = aux[i]
                i += 1

    def _sort(a, lo, hi):
        if hi <= lo:
            return
        mid = lo + (hi - lo) // 2
        _sort(a, lo, mid)
        _sort(a, mid + 1, hi)
        merge(a, lo, mid, hi)
        return a

    a = _sort(a, 0, len(a) - 1)

    return a

#_a = np.random.randint(-999999,999999,10)
_a = np.random.randint(-10,10,16)
print(merge_sort((_a)))