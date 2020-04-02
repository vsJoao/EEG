import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class Epochs:
    def __init__(self, X, classe):
        if np.size(np.shape(X)) != 3:
            print('A matriz não possui a forma [N,T,E] requerida')

        else:
            self.data = X
            self.classe = classe
            self.filtered = dict()

    def filt(self, f_bank: dict, fs: int):
        for f_int in f_bank:
            sos = signal.iirfilter(N=6, Wn=f_bank[f_int], rs=20, btype='bandpass',
                                   output='sos', fs=fs, ftype='cheby2')
            self.filtered['{}-{}'.format(f_bank[f_int][0], f_bank[f_int][1])] = \
                signal.sosfilt(sos, self.data, axis=1)


fb_freqs = {
    1: [8, 12],
    2: [12, 16],
    3: [16, 20],
    4: [20, 24],
    5: [24, 28],
    6: [28, 32]
}

data = np.load('epoch_train/A01T_epoch.npy', allow_pickle=True).item()

epc = {'l': Epochs(data['l'], 'l'),
       'r': Epochs(data['r'], 'r'),
       'f': Epochs(data['f'], 'f'),
       't': Epochs(data['t'], 't')}

epoca = epc['r']
epoca.filt(f_bank=fb_freqs, fs=250)

plt.figure(1)
for i in range(22):
    plt.plot(epoca.data[i, :, 0] - 30 * i)


for j_cnt, j in enumerate(epoca.filtered):
    plt.figure(j_cnt + 2)
    for i in range(22):
        plt.plot(epoca.filtered[j][i, :, 0] - 10 * i)

plt.show()

print('fim')

















