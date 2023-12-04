import cupy as cp
from obspy import Stream

class SeismicCupyArray(cp.ndarray):
    def __init__(self, stream):
        """Converts an obspy Stream to a cupy array preserving metadata.
        """
        msg = "Must initialize with an obspy.Stream object."
        assert isinstance(stream, Stream), msg

        self.array = stream_to_cparray(stream)
        self.stats = [tr.stats for tr in stream]

def stream_to_cparray(stream):
    return cp.stack([cp.asarray(trace.data) for trace in stream.traces])
    # add metadata perhaps, would need to be a subclass of cupy.ndarray though