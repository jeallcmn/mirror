import Graph
from Signal import Signal
from Wave import Wave
from Record import Record
import numpy as np

def sweep(args):
    s = Signal(sample_rate   = args.sample_rate,
            duration      = args.duration,
            amplitude     = args.amplitude,
            start_silence = args.start_silence,
            end_silence   = args.end_silence,
            sweep_range   = [args.freq_min, args.freq_max]
    )
    return s

def record(args):
    signal = sweep(args)
    recorded = Record.record(signal, args.input_device or args.device, args.output_device or args.device)
    normalized = Wave.normalize(recorded)
    Wave.save(args.filename, signal.sample_rate, normalized)   

def play(args):
    signal = sweep(args)
    Record.play(signal, args.output_device or args.device)

def deconvolve(args):
    recording = Wave.read(args.recording)
    _ir(args, recording.data)

def _ir(args, data):
    signal = sweep(args)

    print(data)
    ir = signal.get_impulse(data)

    ir = Wave.normalize(ir)
    if args.trim_begin:
        ir = Wave.trim_begin(ir)
    if args.length:
        ir = Wave.trim_length(args.sample_rate, ir, args.length)
    if args.trim_end:
        ir = Wave.trim_end(ir)
    if args.align_phase:
        ir = Wave.align_phase(ir)
    Wave.save(args.filename, signal.sample_rate, ir)   
    
def ir(args):
    signal = sweep(args)
    print(f"Sweep: {signal.sin_sweep.shape}")
    recorded = Record.record(signal, args.input_device or args.device, args.output_device or args.device)

    normalized = Wave.normalize(recorded)
    print(f"Normalized: {normalized.shape}")

    _ir(args, normalized[:,0])

def generate(args):
    s = sweep(args)
    
    Wave.save( filename = args.filename, 
            sample_rate = args.sample_rate, 
            data = s.sin_sweep )

def convolve(args):
    s = sweep(args)
    ir = Wave.read(filename = args.ir)
    output = s.apply_impulse(ir.data)
    Wave.save( filename = args.filename, sample_rate = args.sample_rate, data = output)

def graph(args):
    ir = [Wave.read(filename = args.irs[0])]
    if len(args.irs) > 1:
        ir.append( Wave.read(filename = args.irs[1]))
    print(f"Graphing {args.irs} with {args.num_samples} samples...")
    Graph.display_ir(ir, args.num_samples)
