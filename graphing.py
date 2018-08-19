import loadSPEfiles as lf
import numpy as np
from matplotlib import pyplot as plt
from os import path as path
import time
import smoothen


def plot_all(x, y):
    """
    Draws a noise-cancelled function using the Savitzky Golay function
    :param x: List of x-axis points
    :param y: List of y-axis points
    :return: N/A
    """
    yprime = smoothen.savitzky_golay(y, 31, 10, 0)
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
    print('find_local_maxes(): ' + str(round((time.time() - starttime) * 1000, 4)) + ' ms')
    return maxes


def find_local_maxes_ws2(wavelength, intensity, pradiusnm=40, ws2c = 450, ws2b = 525, ws2a = 625):
    """
    Finds local maxes for WS2 spectra by first assuming the initial location of the maxes.
    :param wavelength: Numpy array of wavelengths, in nanometers
    :param intensity: Numpy array of intensities
    :param pradiusnm: a radius of pradius nanometers around an assumed position where a peak will be
    :param ws2c:
    :param ws2b:
    :param ws2a:
    :return: a numpy array of peaks
    """
    starttime = time.time()
    maxes = np.array([0])
    # Assumed positions for peaks:


    start_index = find_index(wavelength, ws2c - pradiusnm if ws2c - pradiusnm > ws2c else ws2c)    # where to start searching for a peak (index)
    stop_index = find_index(wavelength, ws2c + pradiusnm)     # where to stop searching for a peak (index)
    area = wavelengths[start_index:stop_index]
    locus = intensity[start_index:stop_index]                 # array that represents where the peak is
    maxes = np.append(maxes, np.argmax(locus) + start_index)
    # maxes = np.append(maxes, find_local_maxes(area, locus, 60) + start_index)
    # find the index of the peak in the new array and add it to the start index, and add it all to maxes

    start_index = find_index(wavelength, ws2b - pradiusnm)
    stop_index = find_index(wavelength, ws2b + pradiusnm)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, np.argmax(locus) + start_index)

    start_index = find_index(wavelength, ws2a - pradiusnm)
    stop_index = find_index(wavelength, ws2a + pradiusnm)
    locus = intensity[start_index:stop_index]
    maxes = np.append(maxes, np.argmax(locus) + start_index)

    maxes = np.delete(maxes, [0])
    print('find_local_maxes_ws2(): ' + str(maxes))
    print('find_local_maxes_ws2(): ' + str(round((time.time() - starttime) * 1000, 4)) + ' ms')
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
    maxes = find_local_maxes_ws2(wavelength, smoothen.savitzky_golay(intensity, 31, 10, 0))
    return np.array([maxes[1]-maxes[0], maxes[2]-maxes[1], maxes[2]-maxes[0]])


if __name__ == '__main__':
    data = lf.load(path.join('Data\WS2 reflection spectra[130]\WS2 reflection spectra', '20180404 WS2_1 a.spe'))
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

    plot_all(wavelengths, intensities)
    #print('Distances: ' + str(find_exciton_peak_distance_ws2(wavelengths, intensities)))
    plt.show()
