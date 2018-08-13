import loadSPEfiles as lf
import numpy as np
from matplotlib import pyplot as plt
from os import path as path
import time


def plot_all(x, y):
    """
    Draws a noise-cancelled function using the Savitzky Golay function
    :param x: List of x-axis points
    :param y: List of y-axis points
    :return: N/A
    """
    yprime = savitzky_golay(y, 31, 10, 0)
    plot_annotate(x, y, graph_1, np.ndarray.tolist(find_local_maxes(x, y)),
                  plot_name='LMaxNorm')
    plot_annotate(x, yprime, graph_2, np.ndarray.tolist(find_local_maxes(x, yprime)),
                  plot_name='LMaxSG', color='r')
    plot_annotate(x, yprime, graph_3, np.ndarray.tolist(find_local_maxes_ws2(x, yprime)),
                  plot_name='LMaxSG_WS2', color='r')


def plot_annotate(x, y, graph_location, maxes, plot_name='Unnamed', color='b'):
    """
    Plot with a certain color and markers, then annotates
    :param x: List of x-axis points
    :param y: List of y-axis points
    :param graph_location: Which graph this plot should show up on
    :param maxes: List of max values in the plot (x, y)
    :param plot_name: The name of the plot for annotations
    :param color: Color to use when plotting
    :return: N/A
    """
    graph_location.plot(x, y, marker='v', markevery=maxes, color=color)
    for r in range(0, len(maxes)):
        graph_location.annotate(plot_name + ': (' + (str(round(x[maxes[r]], 3)) + ', ' + str(round(y[maxes[r]], 3)))
                                + ')', xy=(x[maxes[r]], y[maxes[r]]))


def find_local_maxes(x, y, pradius=200):
    """
    Finds relative maxes in the list of y points.
    :param x: List of x-axis points
    :param y: List of y-axis points
    :param pradius: A peak will occur in +-pradius of a certain point.
    :return: A numpy array of each max's x-axis point
    """
    starttime = time.time()
    maxes = np.array([0])
    peak_radius = pradius       # checks if this is the greatest value in 200 data points [~20 nanometers]
    r = 200
    while r < len(x):
        if y[r] < y[r - peak_radius] and y[r - peak_radius] > y[r - peak_radius * 2]:
            
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
                maxes = np.append(maxes, r - peak_radius)
                r += peak_radius     # continue looking again after the edge of the peak radius
            else:
                r += 1
        else:
            r += 1
    maxes = np.delete(maxes, [0])
    print('find_local_maxes(): ' + str(maxes))
    print('find_local_maxes(): ' + str(time.time() - starttime) + ' ms')
    return maxes


def find_local_maxes_ws2(wavelength, intensity, pradiusnm=50):
    """
    Finds local maxes for WS2 spectra by first assuming the initial location of the maxes.
    :param wavelength: Numpy array of wavelengths, in nanometers
    :param intensity: Numpy array of intensities
    :param pradiusnm: a radius of pradius nanometers around an assumed position where a peak will be
    :return: a numpy array of peaks
    """
    starttime = time.time()
    maxes = np.array([0])
    # Assumed positions for peaks:
    ws2c = 450
    ws2b = 525
    ws2a = 625

    start_index = find_index(wavelength, ws2c - pradiusnm)    # where to start searching for a peak (index)
    stop_index = find_index(wavelength, ws2c + pradiusnm)     # where to stop searching for a peak (index)
    locus = intensity[start_index:stop_index]               # array that represents where the peak is
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)
    # find the index of the peak in the new array and add it to the start index, and add it all to maxes

    start_index = find_index(wavelength, ws2b - pradiusnm)
    stop_index = find_index(wavelength, ws2b + pradiusnm)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)

    start_index = find_index(wavelength, ws2a - pradiusnm)
    stop_index = find_index(wavelength, ws2a + pradiusnm)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)

    maxes = np.delete(maxes, [0])
    print('find_local_maxes_ws2(): ' + str(maxes))
    print('find_local_maxes_ws2(): ' + str(time.time() - starttime) + ' ms')
    return maxes


def find_local_maxes_wse2(wavelength, intensity, pradius=20):
    """
    Finds local maxes for WSE2 spectra by first assuming the initial location of the maxes.
    :param wavelength: Numpy array of wavelengths, in nanometers
    :param intensity: Numpy array of intensities
    :param pradius: a radius of pradius nanometers around an assumed position where a peak will be
    :return: a numpy array of peaks
    """
    maxes = np.array([0])
    # Assumed positions for peaks:
    wse2bp = 485
    wse2ap = 575
    wse2b = 610
    wse2a = 760
    
    start_index = find_index(wavelength, wse2bp - pradius)
    stop_index = find_index(wavelength, wse2bp + pradius)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)

    start_index = find_index(wavelength, wse2ap - pradius)
    stop_index = find_index(wavelength, wse2ap + pradius)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)

    start_index = find_index(wavelength, wse2b - pradius)
    stop_index = find_index(wavelength, wse2b + pradius)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)

    start_index = find_index(wavelength, wse2a - pradius)
    stop_index = find_index(wavelength, wse2a + pradius)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, find_index(locus, np.amax(locus), exact=True) + start_index - 1)

    maxes = np.delete(maxes, [0])
    print('find_local_maxes_wse2(): ' + str(maxes))
    return maxes


def find_index(x, point, exact=False):
    """
    Finds the index of a certain wavelength using the np.where() function
    :param x: Array of wavelengths
    :param point: The wavelength to search for
    :param exact: Whether an exact value is being searched for (if not, a rounded value will be searched)
    :return: the index of the wavelength
    """
    if exact:
        r = np.where(x == point)[0]
    else:
        r = np.where(np.round(x, decimals=0) == round(point))[0]
    # print("Where" + str(r))
    return int(np.average(r))


def find_exciton_peak_distance_ws2(wavelength, intensity):
    """
    Method to find distance between exciton peaks
    :param wavelength: Numpy array of wavelengths, in nanometers
    :param intensity: Numpy array of intensities
    :return: A numpy array of distances in nm where [0] is the distance between C and B, [1] is the distance between
    B and A, and [2] is the distance between A and C
    """
    maxes = find_local_maxes_ws2(wavelength, savitzky_golay(intensity, 31, 10, 0))
    return np.array([maxes[1]-maxes[0], maxes[2]-maxes[1], maxes[2]-maxes[0]])


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
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


if __name__ == '__main__':
    data = lf.load(path.join('Data\WS2 reflection spectra[130]\WS2 reflection spectra', '20180420 WS2_2 a.spe'))
    wavelengths = data[0]
    intensities = data[1]

    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)
    '''
    print('Wavelength\t\t\t\tIntensity')
    for i in range(0, wavelengths.size):
        print(str(wavelengths[i]) + '\t\t' + str(intensities[i]))
    '''
    figure, (graph_1, graph_2, graph_3) = plt.subplots(1, 3, sharey=True)
    figure.suptitle('Spectra')

    print('Distances: ' + str(find_exciton_peak_distance_ws2(wavelengths, intensities)))
    plot_all(wavelengths, intensities)
    plt.show()
