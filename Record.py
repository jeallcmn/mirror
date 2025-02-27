# import sounddevice as sd
import pyaudio
import sys
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from Wave import Wave

class CallbackHandler:
    def __init__(self, data):
        self.current_frame = 0
        self.data = data
        self.buffer = bytearray()
    def callback(self, in_data, frames, time, status):
        if in_data != None:
            # buffer = np.frombuffer(in_data, dtype=np.float32)
            self.buffer.extend(in_data)
            # self.recorded = np.append(self.recorded, buffer)
        if status:
            print(status)
        chunksize = min(len(self.data) - self.current_frame, frames)
        chunk = self.data[self.current_frame:self.current_frame + chunksize].astype(np.float32).tobytes()
        self.current_frame += chunksize
        return (chunk, pyaudio.paContinue)
    def recorded(self):
        return np.frombuffer(self.buffer, dtype=np.float32)
    
class Record:
    def __init__(self):
        pass

    def set_defaults(sample_rate, id, od):
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sd.default.dtype = np.int16
        sd.default.blocksize = 32768
        sd.default.device = (id, od)

    def play(signal, output_device):
            Record.set_defaults(signal.sample_rate, output_device,output_device)
            data = Wave.convertToInt(signal.sin_sweep, np.int16)
            print(f"Play wav data {data.dtype}, {signal.sample_rate}")
    
            sd.play(data, signal.sample_rate, blocking = False)

            sd.wait()


    def record(signal, input_device, output_device):
        Record.set_defaults(signal.sample_rate, input_device, output_device)
        
        data = Wave.convertToInt(signal.sin_sweep, np.int16)
        print(f"Play wav data {data.dtype}, {signal.sample_rate}")

        recorded = sd.playrec(data, signal.sample_rate, channels=1, blocking = False)
        sd.wait()
        return recorded
    
