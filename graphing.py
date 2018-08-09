import loadSPEfiles as lf
import numpy as np
from matplotlib import pyplot as plt
from os import path as path
import math as Math


def draw_normal_plot(x, y):
    '''
    Draws a normal plot. Make sure x and y are numpy arrays of the same size
    :param x: List of x-axis points
    :param y: List of y-axis points
    :return: N/A
    '''
    plt.figure(1)
    maxs = np.ndarray.tolist(find_local_maxs(x, y))
    plt.plot(x, y, marker='v', markevery=maxs)
    plt.show(block=False)


def draw_savitzky_golay(x, y):
    '''
    Draws a noise-cancelled function using the Savitzky Golay function
    :param x: List of x-axis points
    :param y: List of y-axis points
    :return: N/A
    '''
    yprime = savitzky_golay(y, 31, 10, 0)
    plt.figure(4)
    maxs = np.ndarray.tolist(find_local_maxs_ws2(x, yprime, 35))
    plt.plot(x, yprime, marker='v', markevery=maxs)
    plt.show()


def find_local_maxs(x, y, pradius=200):
    '''
    Finds relative maxs in the list of y points.
    :param x: List of x-axis points
    :param y: List of y-axis points
    :param pradius: A peak will occur in +-pradius of a certain point.
    :return: A numpy array of each max's x-axis point
    '''
    maxs = np.array([0]) 
    # peak_radius and skip assume the shape of the data. though they're not terrible assumptions,
    #       in this case we can assume something about the underlying physics. [Change made - now an argument]
    # since you're looking for distance btwn A and B exciton peaks, you can assume that the peaks
    #       will be within some range of their known positions.
    # maybe start from there, instead of assuming that the peaks are >20 nm apart
    peak_radius = pradius       # checks if this is the greatest value in 200 data points [~20 nanometers]
    r = 200
    while r < len(x):
        if y[r] < y[r - peak_radius] and y[r - peak_radius] > y[r - peak_radius * 2]:
            # (r - peak_radius) will be a negative number while r<200. [Doesn't give runtime error??]
            # this will start indexing from the end of y. maybe take the absolute value of the difference? 
            
            # check out https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.amax.html#numpy.amax 
            
            c = r - peak_radius + 1
            add_as_max = True
            while c < r:    # go through the next 200 data points to make sure there isn't a value higher than this.
                if y[c] > y[r - peak_radius]:
                    add_as_max = False  # if there is a higher value, break the loop and continue searching
                    break
                else:
                    c += 1  # continue looking until it reaches r
            if add_as_max:
                maxs = np.append(maxs, r - peak_radius)
                r += peak_radius     # continue looking again after the edge of the peak radius
            else:
                r += 1
        else:
            r += 1
    maxs = np.delete(maxs, [0])
    print(maxs)
    return maxs


def find_local_maxs_ws2(x, y, pradius):
    '''
    Finds local maxes for WS2 spectra by first assuming the initial location of the maxes.
    :param x: Wavelengths, in nanometers
    :param y: Intensities
    :param pradius: a radius of pradius nanometers around an assumed position where a peak will be
    :return: a numpy array of peaks
    '''
    maxs = np.array([0])
    # Assumed positions for peaks:
    ws2c = 450
    ws2b = 525
    ws2a = 625
    temp_peak = 0
    for i in range(find_index(x, ws2c - pradius), find_index(x, ws2c + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    temp_peak = 0
    for i in range(find_index(x, ws2b - pradius), find_index(x, ws2b + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    temp_peak = 0
    for i in range(find_index(x, ws2a - pradius), find_index(x, ws2a + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    maxs = np.delete(maxs, [0])
    print(maxs)
    return maxs


def find_local_maxs_wse2(x, y, pradius):
    '''
    Finds local maxes for WSE2 spectra by first assuming the initial location of the maxes.
    :param x: Wavelengths, in nanometers
    :param y: Intensities
    :param pradius: a radius of pradius nanometers around an assumed position where a peak will be
    :return: a numpy array of peaks
    '''
    maxs = np.array([0])
    # Assumed positions for peaks:
    wse2bp = 485
    wse2ap = 575
    wse2b = 610
    wse2a = 760
    temp_peak = 0
    for i in range(find_index(x, wse2bp - pradius), find_index(x, wse2bp + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    temp_peak = 0
    for i in range(find_index(x, wse2ap - pradius), find_index(x, wse2ap + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    temp_peak = 0
    for i in range(find_index(x, wse2b - pradius), find_index(x, wse2b + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    temp_peak = 0
    for i in range(find_index(x, wse2a - pradius), find_index(x, wse2a + pradius)):
        if y[i] > temp_peak:
            temp_peak = y[i]
            peak_location = i
    maxs = np.append(maxs, peak_location)
    maxs = np.delete(maxs, [0])
    print(maxs)
    return maxs


def find_index(x, wvlth):
    '''
    Finds the index of a certain wavelength
    :param x: Array of wavelengths
    :param wvlth:The wavelength to search for
    :return: the index of the wavelength
    '''
    for xpos in range(0, len(x)):
        if Math.trunc(x[xpos]) == wvlth:
            return xpos


# http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


if __name__ == '__main__':
    data = lf.load(path.join('Data\WS2 reflection spectra[130]\WS2 reflection spectra', '20180423 WS2_2.spe'))
    wavelengths = data[0]
    intensities = data[1]

    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)

    print('Wavelength\t\t\t\tIntensity')
    for i in range(0, wavelengths.size):
        print(str(wavelengths[i]) + '\t\t' + str(intensities[i]))
    draw_normal_plot(wavelengths, intensities)
    draw_savitzky_golay(wavelengths, intensities)