import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import fftconvolve, find_peaks
from Wave import Wave

class Signal():
    minimum_freq = 40
    maximum_freq = 20000
    
    # def __init__(self, sample_rate, duration, amplitude, start_silence, end_silence, sweep_range):
    #     N = int(sample_rate * duration)
    #     sweep = amplitude * Signal.lchirp(N, 0, duration, sweep_range[0], sweep_range[1], cos=False, zero_phase_tmin=False)
    #     invfilter = np.flipud(sweep)
    #     # Add silence at start and end
    #     start = np.zeros(shape = ( start_silence * sample_rate, 1 ) )
    #     end = np.zeros(shape = ( end_silence * sample_rate, 1 ) )

    #     sweep = np.expand_dims(sweep,axis = 1)
    #     self.sin_sweep = np.concatenate((start, sweep, end), axis=0)
    #     self.inverseFilter = invfilter
    #     self.sample_rate = sample_rate

    def __init__(self, sample_rate, duration, amplitude, start_silence, end_silence, sweep_range):
        f1 = np.max([sweep_range[0], Signal.minimum_freq])
        f2 = np.min([sweep_range[1], Signal.maximum_freq, int(sample_rate/2)])
        

        

        w1 = 2 * np.pi * f1 / sample_rate
        w2 = 2 * np.pi * f2 / sample_rate
        num_samples = duration * sample_rate
        sin_sweep = np.zeros(shape = (num_samples, 1))

        self.time_axis = np.arange(0, num_samples,1)/(num_samples -1)
        lw = np.log(w2/w1)
        
        # Expontial sine sweep
        sin_sweep = amplitude * np.sin(w1 * (num_samples-1) / lw * (np.exp(self.time_axis * lw) - 1))

        # Find the last zero crossing to avoid the need for fadeout
        k = np.flipud(sin_sweep)
        error = 1
        counter = 0
        while error > 0.001:
            error = np.abs(k[counter])
            counter = counter+1

        k = k[counter::]
        sinsweep_hat = np.flipud(k)
        sin_sweep = np.zeros(shape = (num_samples,))
        sin_sweep[0:sinsweep_hat.shape[0]] = sinsweep_hat


        # the convolutional inverse
        # Holters2009, Eq.(9)
        self.envelope = (w2/w1) ** (-self.time_axis);

        # Holters2009, Eq.10        
        invfilter = np.flipud(sin_sweep) * self.envelope

        scaling = np.pi*num_samples*(w1/w2-1)/(2*(w2-w1)*np.log(w1/w2))*(w2-w1)/np.pi; 

        # 
        sin_sweep = np.expand_dims(sin_sweep,axis = 1)
        # Add silence at start and end
        start = np.zeros(shape = ( start_silence * sample_rate, 1 ) )
        end = np.zeros(shape = ( end_silence * sample_rate, 1 ) )

        self.sin_sweep = np.concatenate((start, sin_sweep, end), axis=0)

        # Set the attributes
        self.Lp = (start_silence + end_silence + duration) * sample_rate;
        self.inverseFilter = invfilter/amplitude**2/scaling
        self.sample_rate = sample_rate


    def get_impulse(self, recording):
        ir = fftconvolve(self.inverseFilter, recording, mode='valid')
        return ir
    
    def apply_impulse(self, ir):
        source = self.sin_sweep[:,0]
        print(f"Applying Impulse, ir shape:{ir.shape}, signal: {source.shape}")
        output = fftconvolve(source, ir, mode='full')
        return output

    def _lchirp(N, tmin=0, tmax=1, fmin=0, fmax=None):
        fmax = fmax if fmax is not None else N / 2
        t = np.linspace(tmin, tmax, N, endpoint=True)

        a = (fmin - fmax) / (tmin - tmax)
        b = (fmin*tmax - fmax*tmin) / (tmax - tmin)

        phi = (a/2)*(t**2 - tmin**2) + b*(t - tmin)
        phi *= (2*np.pi)
        return phi

    def lchirp(N, tmin=0, tmax=1, fmin=0, fmax=None, zero_phase_tmin=True, cos=True):
        phi = Signal._lchirp(N, tmin, tmax, fmin, fmax)
        if zero_phase_tmin:
            phi *= ( (phi[-1] - phi[-1] % (2*np.pi)) / phi[-1] )
        else:
            phi -= (phi[-1] % (2*np.pi))
        fn = np.cos if cos else np.sin
        return fn(phi)
