import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline
from scipy.signal import convolve2d

x = np.arange(0, 5, 1)
y = np.arange(0, 5, 1)
xx, yy = np.meshgrid(x, y)
# z = np.sin(xx**2 + 2.*yy**2)
z = np.zeros([5, 5])
z[::2, ::2] = np.arange(9).reshape([3,3])
f = interp2d(x, y, z, kind='linear')


kernel = np.zeros([3,3])
kernel[::2, 1::2] = 1
kernel[1::2, ::2] = 1
kernel = kernel / 4

res = convolve2d(z, kernel, boundary='symm', mode='same')
res[::2, ::2] = z[::2, ::2]


print(x)
print(y)
print(f)