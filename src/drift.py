import numpy as np

def population_stability_index(expected, actual, bins=10):
    # expected/actual are 1D arrays for one feature; PSI per feature
    expected_perc, _ = np.histogram(expected, bins=bins, range=(min(expected.min(), actual.min()),
                                                                max(expected.max(), actual.max())), density=True)
    actual_perc, _   = np.histogram(actual,   bins=bins, range=(min(expected.min(), actual.min()),
                                                                max(expected.max(), actual.max())), density=True)
    # avoid div-by-zero
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-6, actual_perc)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)
