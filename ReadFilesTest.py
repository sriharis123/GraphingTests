import loadSPEfiles as lf
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lfilter


def drawScatterGraph(x, y):
    plt.scatter(x, y, s=1, alpha=0.10, marker='o')
    plt.show()

def drawNoiseCancelling(x, y, n):
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, y)
    plt.plot(x, yy, linewidth=2, linestyle="-", c="b")
    plt.show()


if __name__ == '__main__':
    data = lf.load('Data\WS2 reflection spectra[130]\WS2 reflection spectra\\20180404 WS2_1 d.spe')

    wavelengths = data[0]
    intensities = data[1]

    print('Wavelength\t\t\t\tIntensity')
    for i in range(0, wavelengths.size):
        print(str(wavelengths[i]) + '\t\t' + str(intensities[i]))
    drawScatterGraph(wavelengths, intensities)
    drawNoiseCancelling(wavelengths, intensities, 20)
