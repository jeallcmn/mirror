import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import fftconvolve, find_peaks
from Wave import Wave

class Signal():
    minimum_freq = 40
    maximum_freq = 20000

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
         
    def record(args):
        print(f"record: {args}")

    def ir(args):
        signal = Signal.sweep(args)
        recording = Wave.read(args.recording)
        ir = signal.get_impulse(recording.data)

        ir = Wave.normalize(ir)
        if args.trim_begin:
            ir = Wave.trim_begin(ir)
        if args.length:
            ir = Wave.trim_length(args.sample_rate, ir, args.length)
        if args.trim_end:
            ir = Wave.trim_end(ir)
        if args.align_phase:
            ir = Wave.align_phase(ir)
        Wave.save(args.ir, signal.sample_rate, ir)   

    def sweep(args):
        s = Signal(sample_rate   = args.sample_rate,
               duration      = args.duration,
               amplitude     = args.amplitude,
               start_silence = args.start_silence,
               end_silence   = args.end_silence,
               sweep_range   = [args.freq_min, args.freq_max]
        )
        return s

    def generate(args):
        s = Signal.sweep(args)
        
        Wave.save( filename = args.filename, 
                sample_rate = args.sample_rate, 
                data = s.sin_sweep )
        if args.graph:
            s.plot()
    def convolve(args):
        s = Signal.sweep(args)
        ir = Wave.read(filename = args.ir)
        output = s.apply_impulse(ir.data)
        Wave.save( filename = args.output, sample_rate = args.sample_rate, data = output)


