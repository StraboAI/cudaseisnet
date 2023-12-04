# GPU and functional implementions of covseisnet algorithms

# Operate on cupy arrays
# There is one transform to pytorch in order to do
# batch computations of eigenvalues. CuPy doesn't do this
# yet AFAIK

import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupy.linalg import eigvalsh, eigh
import torch

def coherence(X, type, eps=1e-10):
    """Get coherences
    """
    eigenvals = calculate_eigenvalues(X, norm="sum")

    if type == "entropy":
        log_eigenvals = cp.log(eigenvals + eps)
        coherence_ = -X.sum(eigenvals * log_eigenvals, axis=-1)

    elif type == "spectral width":
        indices = cp.arange(X.shape[-1])
        coherence_ =  cp.multiply(eigenvals, indices).sum(axis=-1)

    else:
        raise ValueError(
            f"'{type}' is not a valid choice for 'type'. \n \
            Must be one of 'spectral width' or 'entropy'."
        )

    return coherence_

def calculate_eigenvalues(X, norm="sum"):
    matrices = flatten(X)
    # convert to pytorch tensor (doesn't copy array!) to run batch eigvalsh
    matrices = torch.as_tensor(matrices, device="cuda")
    eigenvals = torch.abs(torch.linalg.eigvalsh(matrices))
    eigenvals = torch.flip(eigenvals, dims=(-1,))
    print(eigenvals.shape)
    if norm == "sum":
        denom = eigenvals.sum(1).unsqueeze(1)
    elif norm == "max":
        denom = eigenvals.max(1).unsqueeze(1)
    else:
        raise ValueError(f"'{norm}' is not a choice.")
    eigenvals = eigenvals / denom
    eigenvals = cp.asarray(eigenvals)
    return eigenvals.reshape(X.shape[:-1])

# not done yet - needed?
def calculate_eigenvectors(X, rank=0, covariance=False):
    # Initialization
    matrices = flatten(X)
    eigenvectors = cp.zeros((matrices.shape[0], matrices.shape[-1]), dtype=complex)

    # Calculation over submatrices
    for i, m in enumerate(matrices):
        eigenvectors[i] = eigh(m)[1][:, -1 - rank]

    if covariance:
        raise NotImplemented("Covariance method not yet implemented.")
        ec = cp.zeros(X.shape, dtype=complex)
        ec = flatten(ec)
        for i in range(eigenvectors.shape[0]):
            ec[i] = eigenvectors[i, :, None] * cp.conj(eigenvectors[i])
        ec = ec.reshape(X.shape)
        return ec
    else:
        return eigenvectors.reshape(X.shape[:-1])


def flatten(array):
    return array.reshape(-1, *array.shape[-2:])

def triu(X, **kwargs):
    """Get the upper tringular part of flattened matrix
    """

    trii, trij = cp.triu_indices(X.shape[-1], **kwargs)
    return X[..., trii, trij]

def sliding_windows(X, winlen, step):
    """Creates sliding windows across second dimension of an array
    """
    view = _sliding_window_view(X, winlen, axis=1)[:,::step,...]
    return cp.asarray(view)

def calculate(X, window_duration_sec, average, sr, average_step=None, **kwargs):
    """Calculate covariances from an array"""
    times, frequencies, spectra = stft(X, window_duration_sec, sr, **kwargs)

    # Parametrization
    step = average // 2 if average_step is None else average * average_step
    n_traces, n_windows, n_frequencies = spectra.shape

    # Times
    t_end = times[-1]
    times = times[:-1]
    times = times[: 1 - average : step]
    n_average = len(times)
    times = cp.hstack((times, t_end))

    # Initialization
    cov_shape = (n_average, n_traces, n_traces, n_frequencies)
    covariance = cp.zeros(cov_shape, dtype=complex)
    # Compute - can this be parallelized?
    for t in range(n_average):
        covariance[t] = xcov(t, spectra, step, average)

    frequencies_plotting = cp.linspace(0, sr, len(frequencies) + 1)

    return (
        times,
        frequencies_plotting,
        covariance.transpose([0, -1, 1, 2]),
    )


def stft(
    X,
    window_duration_sec,
    sr,
    bandwidth=None,
    window_step_sec=None,
    window=cp.hanning,
    **kwargs
):
    """Short time Fourier Transform 
    """
    # Time vector - calculate from sr and npts instead of times
    npts = int(window_duration_sec * sr)
    step = npts // 2 if window_step_sec is None else int(window_step_sec * sr)
    times = np.arange(X.shape[-1])[:: step]
    n_times = len(times)
    # Frequency vector
    kwargs.setdefault("n", 2 * npts - 1)
    frequencies = cp.linspace(0, sr, kwargs["n"])

    if bandwidth is not None:
        fin = (frequencies >= bandwidth[0]) & (frequencies <= bandwidth[1])
    else:
        fin = cp.ones_like(frequencies, dtype=bool)
    frequencies = frequencies[fin]

    stacked_windows = sliding_windows(X, npts, step)

    # apply window to the stacks
    stacked_windows = stacked_windows * window(npts)[cp.newaxis,cp.newaxis,:]
    spectra = cufft.fft(stacked_windows, **kwargs)#[fin]

    return times, frequencies, spectra

# parallelize
def xcov(wid, spectra_full, overlap, average):
    """Calculation of the array covariance matrix from the array data vectors.

    Warning
    -------
    This function is not fully documented yet, as may be reformulated.

    To do
    -----
    Allow to possibly reformulate this part with Einstein convention for
    faster computation, clarity and TensorFlow GPU transparency.

    Arguments
    ---------
    spectra_full: :class:`numpy.ndarray`
        The stream's spectra.

    overlap: int
        The average step.

    average: int
        The number of averaging windows.

    Returns
    -------
    :class:`numpy.ndarray`
        The covariance matrix.
    """
    n_traces, n_windows, n_frequencies = spectra_full.shape
    beg = overlap * wid
    end = beg + average
    spectra = spectra_full[:, beg:end, :].copy()
    x = spectra[:, None, 0, :] * cp.conj(spectra[:, 0, :])
    nswids = list(range(1, average))
    for swid in nswids:
        x += spectra[:, None, swid, :] * cp.conj(spectra[:, swid, :])

    return x

def _sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    """
    from https://github.com/cupy/cupy/blob/main/cupy/lib/stride_tricks.py#L43
    This should be in new versions of cupy

    Create a sliding window view into the array with the given window shape.

    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.


    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional -- not supported
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.

    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.


    See also
    --------
    numpy.lib.stride_tricks.as_strided

    Notes
    --------
    This function is adapted from numpy.lib.stride_tricks.as_strided.

    Examples
    --------
    >>> x = _cupy.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    """
    import cupy as _cupy
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))

    # writeable is not supported:
    if writeable:
        raise NotImplementedError("Writeable views are not supported.")

    # first convert input to array, possibly keeping subclass
    x = _cupy.array(x, copy=False, subok=subok)

    window_shape_array = _cupy.array(window_shape)
    for dim in window_shape_array:
        if dim < 0:
            raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = _cupy._core.internal._normalize_axis_indices(axis, x.ndim)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return cp.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)