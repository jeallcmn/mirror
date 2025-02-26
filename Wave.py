import scipy.io.wavfile as wav
import numpy as np
import sys
from scipy import signal


class Wave:
    
    def __init__(self, sample_rate, data):
        self.sample_rate = sample_rate
        self.data = data

    def normalize(data):
        normalized =  data / np.max(np.abs(data),axis=0)
        # normalized *= 0.707 # -3dB
        return normalized
    

    def save(filename, sample_rate, data):
        output_data = Wave.normalize(data.astype(np.float32))
        wav.write(filename, sample_rate, output_data)

    def read(filename):

        sample_rate, data = wav.read(filename)
        data = data.astype(np.float32)
        length = data.shape[0] / sample_rate
        data = Wave.normalize(data)

        return Wave(sample_rate, data)

    def trim_begin(data, threshold=0.02):
        counter = 0
        while np.abs(data[counter]) < threshold:
            counter = counter+1
        
        print(f"Trimming {counter} samples from wave (sample: {data[counter]})")
        # Trim up to the sample before
        return data[counter::]

    def trim_length(sample_rate, data, duration=0.500):
        l = int(sample_rate * duration)
        print(f"Trimming duration to {l} samples ({duration}s)")
        data = data[:l]
        return data
    
    def trim_end(data, threshold=0.00001):
        data = np.flipud(data) # reverse order
        data = Wave.trim_begin(data, threshold) # trim the beginning
        data = np.flipud(data) # flip it back around
        return data

    def align_phase(data, threshold=0.05):
        counter = 0
        while np.abs(data[counter]) < threshold:
            counter = counter+1

        if data[counter] < 0:
            print(f"Flipping phase");

            flipped = np.zeros(shape = ( data.shape[0], 1 ) )
            for x in range(0, data.shape[0]):
                flipped[x] = data[x] * -1
            data = flipped
        return data
