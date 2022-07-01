import typing as tp
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Kernel


def gpr_time_series_smoothing(time_series: np.array,
                              cnt_iter: int = 15,
                              kernel: tp.Optional[Kernel] = None,
                              method: str = 'linear',
                              return_last_gpr_mean_std: bool = False,
                              remove_convergence_warnings: bool = True) -> tp.Sequence:
    """
    Uses Gauss Process Regression (GPR) for estimating mean and std of time series
     if x_t not in mean_t +- std_t, then x_t treated as outlier and it will be interpolated

    :param time_series:
            Time series to be smoothed
    :param cnt_iter:
            Defines cont of iterations (fit_GPR->find_outliers->interpolate)
            cnt_iter >= 1
    :param kernel:
            Kernel passed to GPR
               - if is None: kernel = WhiteKernel(noise_level_bounds=[0.001, 0.05])
                                    + RBF(length_scale_bounds=[2, len(time_series) / 7])
    :param method:
            Interpolation method passed to pandas.Series.interpolate
    :param return_last_gpr_mean_std:
            Defines whether return mean and std of last iteration GPR fitting with smoothed time series or not.
    :param remove_convergence_warnings:
        Just defines whether to remove sklearn.exceptions.ConvergenseWarning or not
    :return:
            - if return_last_gpr_mean_std == True --> tuple(smoothed_ts, gpr_mean, gpr_std)
            - else --> smoothed_ts
    """
    assert cnt_iter >= 1, f"wrong cnt_iter! got {cnt_iter=}"

    time_series_ = pd.Series(time_series.copy())
    len_ts = len(time_series)

    if kernel is None:
        kernel = WhiteKernel(noise_level_bounds=[0.001, 0.05]) + RBF(length_scale_bounds=[2, len_ts / 7])

    if remove_convergence_warnings:
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for it in range(cnt_iter):
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(np.arange(len_ts)[..., None], time_series_)
        mean, std = model.predict(np.arange(len_ts)[..., None], return_std=True)

        mask = (time_series_ < mean - std) | (time_series_ > mean + std)
        if not mask.any():
            break
        time_series_[mask] = np.nan

        assert not np.isnan(time_series_).all(), f"All time series considered as outlier, got {kernel=},  {time_series=}"
        time_series_.interpolate(method=method, limit_direction='both', inplace=True)

    if return_last_gpr_mean_std:
        return time_series_.values, mean, std
    return time_series_.values
