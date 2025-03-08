from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from pathlib import Path
import sys
from Wave import Wave
def display_ir(wavs, num_samples):

    fig, (plt1, plt2)  = plt.subplots(2, figsize=[10, 8])

    plt.subplots_adjust(hspace=1)
    for x in wavs:
        plt1.plot(x.data[:num_samples], label=x.name)
        w, h = signal.freqz(x.data, worN=4096)
        plt2.plot(w * x.sample_rate / (2 * np.pi), np.abs(h), label=x.name)


    plt1.set_xlabel("Sample")
    plt1.set_ylabel("Magnitude")
    plt1.legend(loc="best", bbox_to_anchor=(1,1))
    plt1.set_title("Impulse")
    plt1.grid(True)

    plt2.set_xlabel('Frequency (Hz)')
    plt2.set_ylabel('Magnitude')
    plt2.set_title("Frequency Response")
    plt2.legend()

    plt2.set_xscale('log')
    plt2.set_xlim([20, 20000])

    plt.show()


if __name__ == "__main__":
    irs = (Wave.read(w) for w in sys.argv[1:])
    display_ir(irs,256)
1