from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist(section="training", offset=0, count=None, ret='xy',
               x_dtype=np.float64, y_dtype=np.int64, path=None):
    """
    Loads MNIST dataset using sklearn's fetch_openml.
    Parameters
    ----------
    section : str
        Either "training" or "testing", depending on which section you want to
        load.
    offset : int
        Skip this many samples.
    count : int or None
        Try to load this many samples. Default is None, which loads until the
        end.
    ret : str
        What information to return. See return values.
    x_dtype : dtype
        Type of samples. If ``np.uint8``, intensities lie in {0, 1, ..., 255}.
        If a float type, then intensities lie in [0.0, 1.0].
    y_dtype : dtype
        Integer type to store labels.
    path : str
        Not used in this version since the data is loaded via sklearn.
    Returns
    -------
    images : ndarray
        Image data of shape ``(N, 28, 28)``, where ``N`` is the number of
        images. Returned if ``ret`` contains ``'x'``.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned if ``ret``
        contains ``'y'``.
    """
    
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', version=1)
    images, labels = mnist['data'], mnist['target'].astype(int)

    # Reshape images to (N, 28, 28)
    images = images.values.reshape(-1, 28, 28).astype(x_dtype)

    # Normalize the images if x_dtype is float type
    if x_dtype in (np.float16, np.float32, np.float64):
        images /= 255.0

    # Split into training and testing
    train_size = 60000
    if section == 'training':
        images, labels = images[:train_size], labels[:train_size]
    elif section == 'testing':
        images, labels = images[train_size:], labels[train_size:]
    else:
        raise ValueError("section must be 'training' or 'testing'")
    
    # Apply offset and count
    if count is None:
        count = len(images) - offset
    
    images = images[offset:offset + count]
    labels = labels[offset:offset + count]

    # Return according to 'ret' parameter
    returns = ()
    if 'x' in ret:
        returns += (images,)
    if 'y' in ret:
        returns += (labels,)

    if len(returns) == 1:
        return returns[0]  # Don't return a tuple of one
    else:
        return returns
