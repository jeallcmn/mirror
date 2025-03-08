import scipy.io.wavfile as wav
import numpy as np
import sys
from scipy import signal
from pathlib import Path


class Wave:
    
    def __init__(self, sample_rate, data, filename=None):
        self.sample_rate = sample_rate
        self.data = data
        self.filename = filename
        self.name = Path(filename).stem

    def normalize(data):
        normalized =  data / np.max(np.abs(data),axis=0)
        normalized *= 0.9 # -3dB
        return normalized
    
    def convertToFloat(data):
        if data.dtype in [np.int32, np.int16]:
            max = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max
        elif data.dtype == np.float32:
            pass
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        else:
            raise Exception(f"Unsupported wav format: {data.dtype}")

        return data
    
    def convertToInt(data, type):
        if data.dtype != type:
            max = np.iinfo(type).max
            data = (data * max).astype(type)
        return data

    # Always save as Wave float32
    def save(filename, sample_rate, data, type=np.float32):
        output_data = Wave.normalize(data)
        output_data = Wave.convertToFloat(data)
        
        wav.write(filename, sample_rate, output_data)

    # Converts to internal float32 representation
    def read(filename):
        sample_rate, data = wav.read(filename)
        data =  Wave.convertToFloat(data)        
        data = Wave.normalize(data)

        return Wave(sample_rate, data, filename=filename)

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
