import numpy as np
import pylab as pl
import scipy.signal as signal

from scipy import fftpack

t = np.arange(0, 0.3, 1 / 20000.0)
x = np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 8 * t) + 3.0)
hx = fftpack.hilbert(x)
hx = np.sqrt(x ** 2 + hx ** 2)
pl.subplot(111)
pl.plot(x, label=u"Carrier")
pl.plot(hx, "r", linewidth=2, label=u"Envelop")
pl.title(u"Hilbert Transform")
pl.legend()
pl.show()