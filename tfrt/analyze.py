"""
Utilities to help analyze an optical system.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def imaging_test(
    get_samples, image_range, batch_count=50, bins=128, verbose=True, display=True
):
    """
    Trace many rays and use a histogram to visualize the result.
    
    Parameters
    ----------
    get_samples : callable
        This function should trace and return a set of 2D samples, presumably where the rays
        strike the imagine plane.  The return should have shape (n, 2).
    cycle_count : int
        It's better to trace a lot of rays.  More than the ray tracer can handle in a single
        batch.  This parameter controlls how many batches of rays are traced (get_samples is
        called this many times).
    image_range : float array of shape (2, 2)
        The range over which to display the histogram.  See plt.hist2d for details.  Typically
        should have the format ((-x, x), (-y, y)), but may also be None, in which case the 
        range is inferred.
    bins : int, optional
        The number of bins to use for the histogram.  See plt.hist2d for details.  Defaults
        to 128.
    verbose : bool, optional
        If True, display messages detailing the tracing progress.
    display : bool, optional
        Defaults to True, in which case this opens a plt window and displays the histogram.
        Otherwise simply returns the histogram without displaying it.
        
    Returns
    -------
    Returns the four outputs from plt.hist2d, look it up for more information about the returns
    h : 2D float np array
        The histogram bins.
    xedges, yedges : 1D float np array
        The edges of hisogram bins.
    image : mpl.collections.QuadMesh
        The object generated to display the histogram in a plt plot.  This return is always
        returned for consistency, but it will be None if display was set to False.
    """
    image_samples = []
    for i in range(batch_count):
        image_samples.append(np.array(get_samples()))
        if verbose:
            print(f"Sampling step {i}/{batch_count}-{100*i/batch_count:.2f}%.")
    image_samples = np.concatenate(image_samples)
    print(f"final sample shape: {image_samples.shape}")
    print(f"total rays traced: {image_samples.shape[0]}")    
    
    if display:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_aspect("equal")
        h, xedges, yedges, image = plt.hist2d(
            image_samples[:,0],
            image_samples[:,1],
            bins=bins,
            range=image_range
        )
    else:
        h, xedges, yedges = np.histogram2d(
            image_samples[:,0],
            image_samples[:,1],
            bins=bins,
            range=image_range
        )
        image=None
    if display:
        plt.show()
    return h, xedges, yedges, image

# -------------------------------------------------------------------------------------------
    
def inner_product(first, second):
    """
    Performs an inner product, as a means of comparing two images.
    """
    first = np.array(first)
    second = np.array(second)
    
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    
    return np.sum(first * second)

# ===========================================================================================

def histogram2D(x, y, value_range, x_bins=100, y_bins=None, dtype=tf.float64):
    """
    Generate a 2D histogram
    
    This code was copied from:
    https://gist.github.com/isentropic/a86effab2c007e86912a50f995cac52b
    
    And it was modified slightly.  This function uses tf instead of np to make the histogram.
    
    I had a tremendous amount of confusion about which index correlates to which dimension.
    After a bunch of testing it seems like the popular convention is the opposite of my
    intuition.  And I have been having this problem before...  So in the tensor that this 
    returns, y is the first index and x is the second.
    
    Parameters
    ----------
    x, y : 1D tensor
        The x and y values of the points to bin.
    value_range : 2x2 array-like
        The limits to use for the bins in the x and y directions
    xbins, ybins : int, optional
        Defaults to 100.  The number of bins to use in each dimension
    dtype : tf.dtype, optional
        Defaults to tf.float64.  The dtype to use for the histogram.
    """
    y_bins = y_bins or x_bins
    x_range = tf.cast(value_range[0], dtype)
    y_range = tf.cast(value_range[1], dtype)

    histy_bins = tf.histogram_fixed_width_bins(y, y_range, nbins=y_bins, dtype=dtype)
    
    H = tf.map_fn(lambda i: tf.histogram_fixed_width(
        x[histy_bins == i],
        x_range,
        nbins=x_bins
    ), tf.range(y_bins))
    return H
    
# ===========================================================================================
    
class DistributionDifferential:
    """
    Compute the difference between two distributions.  Ideal for a differential evolution
    objective function.
    
    This class holds a goal distribution, which can either be a callable or an array that
    describes a density function over a 2D domain.  When the class is called with a set of
    2D points, it returns a positive scalar value that describes how similar the input
    distribution is to the held one.
    
    This class uses the histogram2D function described above, which itself uses 
    tf.histogram_fixed_width, which add all points outside the histogram's domain into the
    first and last bins.  This may not be ideal for optimization, so this class adds the 
    ability to add an additional penalty to points that fall outside the domain of the goal.
    If oob_penalty is None, this step is skipped and points outside the domain will be 
    binned into the edges of the histogram, but if it is not None, it must be a callable that
    accepts a 1D array of distances, and returns some positive value to use as the penalty 
    for falling outside of the domain.  The distance used for this purpose will that between
    each point and the center of the domain.  The penalty will be divided by the number of
    points being penalized to prevent it from being inappropriately large.
    
    I feel like something along the linds of lambda x: a*x*x + b*tf.ones_like(x) is a good
    starting point for a penalty function.  The constants need to be tweaked through
    experimentation.
    """
    def __init__(self, goal, domain, x_bins=50, y_bins=None, oob_penalty=None):
        """
        Parameters
        ----------
        goal : callable or 2D float array
            The ideal distribution to compare to.  If it is a callable, it must accept two
            arrays of points, x and y, and must output an array of the same shape of floats
            greater than zero, which describe the density.  Need not be normalized, and will
            be normalized by this class.  If it is not a callable, it must be a 2D array;
            the density having already been evaluated on the proper grid.
        domain : int array of shape (2, 2)
            The domain to evaluate everything on, in the format
            ((x_start, x_end), (y_start, y_end)).
        x_bins, y_bins: int, optional
            The number grid points to evaluate.  x_bins defaults to 50, y_bins defaults to
            the same value as x_bins.  If goal is not a callable, these parameters are
            ignored.
        oob_penalty : callable or None, optional
            If None, points that fall outside the domain are binned into the edges of the
            domain.  Otherwise, they are filtered off, have their distance to the center of
            the domain computed, and that distance is fed to this function to generate a 
            penalty value for points that fall outside the domain.            
        """
        self._x_bins = x_bins
        self._y_bins = y_bins or x_bins
        
        # error check the domain
        try:
            self._domain = domain
            self._x_start = domain[0][0]
            self._x_end = domain[0][1]
            self._y_start = domain[1][0]
            self._y_end = domain[1][1]
        except(IndexError) as e:
            raise ValueError(
                "DistributionDifferential: domain must have shape (2, 2)."
            ) from e
        
        # check if goal has a shape   
        try:
            if goal.rank != 2:
                raise ValueError(
                    "DistributionDifferential: goal must be 2D."
                )
            else:
                # goal is good, so override x and y bins from its shape
                self._x_bins, self._y_bins = goal.shape
        except(AttributeError):
            # if goal doesn't have  shape, it must be a callable, so generate an
            # evaluation grid and evaluate it.
            
            # more error checking
            try:
                eval_grid_x = tf.linspace(self._x_start, self._x_end, self._x_bins + 1)
                eval_grid_y = tf.linspace(self._y_start, self._y_end, self._y_bins + 1)
            except(TypeError) as e:
                raise TypeError(
                    "DistributionDifferential: bin counts must be ints."
                ) from e
            except(tensorflow.python.framework.errors_impl) as e:
                raise TypeError(
                    "DistributionDifferential: domain values must be floats."
                ) from e
        
            # turn the coords into a grid, centered on the center of the bins
            eval_grid_x = (eval_grid_x[:-1] + eval_grid_x[1:])/2.0
            eval_grid_y = (eval_grid_y[:-1] + eval_grid_y[1:])/2.0
            eval_grid_x, eval_grid_y = tf.meshgrid(eval_grid_x, eval_grid_y)
            self._eval_grid_x = eval_grid_x
            self._eval_grid_y = eval_grid_y
            
            # attempt to evaluate the goal on the grid
            try:
                goal = goal(eval_grid_x, eval_grid_y)
            except Exception:
                raise ValueError(
                    "DistributionDifferential: goal must be a callable that accepts two "
                    "arrays of points, or a 2D array."
                )
        
        # normalize the goal
        self._goal, _ = tf.linalg.normalize(goal)
        
        # check oob_penalty
        self._oob_penalty = oob_penalty
        if oob_penalty:
            try:        
                oob_penalty(tf.zeros(5))
                
            except Exception as e:
                raise ValueError(
                    "DistributionDifferential: oob_penalty must be a callable that accepts "
                    "an array, or None."
                ) from e
                
    def _distance(self, x, y):
        x = x - (self._x_start + self._x_end)/2.0
        y = y - (self._y_start + self._y_end)/2.0
        return tf.sqrt(x*x + y*y)
                
    def __call__(self, x, y):
        if self._oob_penalty:
            # split x and y into two sets, in and out of bounds
            x_low = x < self._x_start
            x_high = x > self._x_end
            y_low = y < self._y_start
            y_high = y > self._y_end
            
            x_oob = tf.math.logical_or(x_low, x_high)
            y_oob = tf.math.logical_or(y_low, y_high)
            oob = x_oob = tf.math.logical_or(x_oob, y_oob)
            not_oob = tf.logical_not(oob)

            penalty = self._oob_penalty(self._distance(
                tf.boolean_mask(x, oob),
                tf.boolean_mask(y, oob)
            ))
            penalty = tf.reduce_sum(penalty / tf.cast(tf.shape(penalty)[0], tf.float64))
            
            x = tf.boolean_mask(x, not_oob)
            y = tf.boolean_mask(y, not_oob)
        
        # obtain a distribution of the in bounds points, and compare to the goal.
        histo = histogram2D(x, y, self._domain, x_bins=self._x_bins, y_bins=self._y_bins)
        histo, _ = tf.linalg.normalize(tf.cast(histo, tf.float64))
        self.saved_histo = histo
        
        quality = tf.reduce_sum(tf.math.squared_difference(histo, self._goal))
        
        if self._oob_penalty:
            return quality + penalty
        else:
            return quality
        
        
    
    
    
    
    
    
    
