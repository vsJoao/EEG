import numpy as np

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

X = np.load('epoch_train/A01T_epoch.npy', allow_pickle=True).item()
y = X['l'][1, :, 0]

x = np.arange(0, np.shape(y)[0]/250, 1/250)
sos = signal.cheby2(N=2, rs=20, Wn=[40, 45], btype='bandpass', output='sos', fs=250)
y_filtered = signal.sosfilt(sos, y)

plt.subplot(211)
plt.plot(x, y)
plt.subplot(212)
plt.plot(x, y_filtered)
plt.show()

