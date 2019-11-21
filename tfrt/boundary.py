"""
Classes that represent optical boundaries.
"""

from abc import ABC, abstractmethod
from tfrt.update import RecursivelyUpdatable

import tensorflow as tf
import numpy as np

# =====================================================================================

class SegmentBoundaryBase(RecursivelyUpdatable):
    """
    Base class for boundaries made out of line segments.
    
    Requires sub classes to implement _update, _generate_update_handles.
    
    catagory defaults to generic, but should commonly be 'optical', 'stop', 'target'.
    Describes what the boundary is used for, but I am not yet sure this will actually be
    used by the tracer.
    """
    
    def __init__(self, name=None, catagory="generic", extra_fields=set(), **kwargs):
        self._name = name
        self._catagory = catagory
        fields = set({"x_start", "y_start", "x_end", "y_end"}) | extra_fields
        self._fields = {f: None for f in fields}
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self._name
        
    @property
    def catagory(self):
        return self._catagory
        
    @property
    def dimension(self):
        return 2
       
    @property 
    def signature(self):
        return set(self._fields.keys())
        
    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, item):
        self._fields[key] = item
        
# -------------------------------------------------------------------------------------
        
class ManualBoundary(SegmentBoundaryBase):
    """
    Class that builds a boundary directly from a set of points.
    
    If update_function is specified, it must be a callable that takes no argument and
    returns four values, x_start, y_start, x_end, y_end.  Defaults to None, and if left
    as None, calling update on this class will do nothing.
    
    segment_splitter is a convenient function for converting from the segment data
    format to the one needed by update_function.
    
    Data can be fed to this class either by indexing its fields after instantiation
    (ManualBoundary["x_start"] = my_points) or by calling feed_segments, which takes
    a list of 4-tuples, each of which is (x_start, y_start, x_end, y_end).
    
    Can add new fields at any time simply by indexing the new field.
    """
    def __init__(self, update_function=None, **kwargs):
        self.update_function = update_function
        super().__init__(**kwargs)
        
    def _generate_update_handles(self):
        return []
            
    def _update(self):
        if self.update_function is not None:
            self["x_start"], self["y_start"], self["x_end"], self["y_end"] = \
                self.update_function()
                
    def feed_segments(self, segments):
        self["x_start"], self["y_start"], self["x_end"], self["y_end"] = \
            self.segment_splitter(segments)
    
    @staticmethod    
    def segment_splitter(segments):
        x_start, y_start, x_end, y_end = tf.unstack(segments, axis=1)
        return tf.reshape(x_start, (-1,)), tf.reshape(y_start, (-1,)), \
            tf.reshape(x_end, (-1,)), tf.reshape(y_end, (-1,))
            
# -------------------------------------------------------------------------------------

class 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
