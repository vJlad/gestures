import numpy as np
import typing as tp


def dp_time_series_distance(a: tp.Sequence,
                            b: tp.Sequence,
                            max_erases: tp.Union[int, tp.Callable[[int, int], int]] = 10,
                            pairwise_distance: tp.Callable = lambda x, y: np.abs(x - y),
                            erase_cost: tp.Callable[[tp.Sequence, int], tp.Any]
                            = lambda x, ind: abs(x[ind + 1] - x[ind]) / 10.0 if ind + 1 < len(x) else 0.0,
                            inf_const: tp.Any = 1e18,
                            dp_dtype: np.dtype = np.float32) -> tp.Any:
    """
    computes distance like longest common subsecuence dp:
        from each sequence can erase any max_erases elements (with shift)
        minimises average pairwise distance between non-erased elements

    dp contains minimal average distance of some state
            state = (a_id, a_shift, b_shift),
            x_id --- index in the sequence x, x={a, b}
            x_shift --- count of used erase operations in the sequence x
            b_id = a_id + b_shift - a_shift
            (not including a_id, b_id)

    :param a: first time series
    :param b: second time series
    :param max_erases: maximal count of erase operations from each of input time series
    :param pairwise_distance: distance between samples from input time series
    :param erase_cost: cost of erasing element from time series
    :param inf_const: const used to initialise dp (very large constant), (inf_const >= dp).all()
    :param dp_dtype: type of elements of dp
    :return: minimal average distance
    """
    len_a = len(a)
    len_b = len(b)
    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    if isinstance(max_erases, int):
        assert max_erases >= 0, \
            f'count of erase operations should be a nonnegative integer or function(int,int)->int, got {max_erases=}'
    else:
        max_erases = max_erases(len_a, len_b)

    assert abs(len_a - len_b) <= max_erases, \
        f'time series are not comparable, try increase max_erases. Got {len(a)=}, {len(b)=}, {max_erases=}'

    dp: np.array = np.ones((len_a + 1, max_erases + 1, max_erases + 1)) * inf_const
    dp = dp.astype(dp_dtype)
    dp[0][0][0] = 0

    for a_id in range(len_a + 1):
        for a_shift in range(max_erases + 1):
            if a_id < a_shift:
                break
            for b_shift in range(max_erases + 1):
                b_id = a_id - a_shift + b_shift
                if a_id < len_a and b_id < len_b:
                    dp[a_id + 1, a_shift, b_shift] = min(dp[a_id + 1, a_shift, b_shift],
                                                         dp[a_id, a_shift, b_shift]
                                                         + pairwise_distance(a[a_id], b[b_id]))

                if a_shift < max_erases and a_id < len_a:
                    dp[a_id + 1, a_shift + 1, b_shift] = min(dp[a_id + 1, a_shift + 1, b_shift],
                                                             dp[a_id, a_shift, b_shift]
                                                             + erase_cost(a, a_id))

                if b_shift < max_erases and b_id < len_b:
                    dp[a_id, a_shift, b_shift + 1] = min(dp[a_id, a_shift, b_shift + 1],
                                                         dp[a_id, a_shift, b_shift]
                                                         + erase_cost(b, b_id))

    len_diff = len_b - len_a
    cnt_usd = len_a - np.arange(max(0, -len_diff),
                                min(max_erases + 1, max_erases - len_diff + 1))
    ans = (dp[len_a,
              np.arange(max(0, -len_diff),
                        min(max_erases + 1, max_erases - len_diff + 1)),
              np.arange(max(0, -len_diff),
                        min(max_erases + 1, max_erases - len_diff + 1))
              + len_diff] / cnt_usd).min()
    return ans
