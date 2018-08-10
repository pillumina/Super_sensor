from packages import *
from utils import *
from scipy import signal

import pywt

dir = 'PIR_C19_NO'

arr = data2arr(dir, False)
arr = elim_outliers(arr, dir, False)

xx = np.arange(len(arr))

(cA, cD) = pywt.dwt(arr, 'db1')

print(len(arr))
print(len(cA))
print(cD[:100])

arr2 = pywt.idwt(cA, cD, 'db1')

plt.subplot(2,2,1)
plt.plot(np.arange(len(arr)), arr)

plt.subplot(2,2,2)
plt.plot(np.arange(len(cA)), cA)

plt.subplot(2,2,3)
plt.plot(np.arange(len(cD)), cD)

plt.subplot(2,2,4)
plt.plot(np.arange(len(arr2)), arr2)
# plt.contour(xx, cD, abs(cA))

plt.show()




