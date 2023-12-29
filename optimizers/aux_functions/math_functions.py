import numpy as np
import sys


def aic(predicted_data: list, test_data: list, k: int):
    print("predicted_data: ", len(predicted_data))
    print("test_data: ", len(test_data))
    if len(predicted_data) == len(test_data):
        n = len(predicted_data)
        sse = sum([(y - fy) ** 2 for y, fy in zip(predicted_data, test_data)])
        sigma = sse / (n - 2)

        aic_criterion = 2 * k + n * np.log(sigma)
        return aic_criterion
    else:
        print(f"Error in module {sys.argv[0].split('/')[-1]}, the dimensions of the input data don't converge")
        return -1

def aic_rss(rss, n, k):
    rss = sum(rss)
    sigma = rss / n  # sse / (n - 2)
    aic_criterion = 2 * k + n * np.log(sigma)
    if k > (n/40):
        print("AIC: ", aic_criterion)
        aic_corrected = aic_criterion + ((2 * (k ** 2) + (2 * k)) / (n - k - 1))
        return aic_corrected
    else:
        return aic_criterion
