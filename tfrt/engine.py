"""
The class that will hold the ray tracer, and the data organization tools it uses.
"""

from abc import ABC, abstractmethod
import tensorflow as tf

from tfrt.update import RecursivelyUpdatable
import tfrt.operation as op

class SetBase(RecursivelyUpdatable):
    """
    Base class for set objects.
    """
    def __init__(self, sources, signature, **kwargs):
        self._signature = signature
        self._fields = {}
        if type(sources) is not list:
            sources = [sources]
        self._sources = sources
        super().__init__(**kwargs)
                
    @property
    def signature(self):
        return self._signature
        
    def _generate_update_handles(self):
        return [source.update for source in self._sources]
        
    def _update(self):
        for sig in self._signature:
            try:
                self._fields[sig] = tf.concat(
                    [source[sig] for source in self._sources],
                    axis=0
                )
            except (KeyError):
                self._fields[sig] = None
   
    def __getitem__(self, key):
        return self._fields[key]
        
    def __setitem__(self, key, item):
        if key in self.signature:
            self._fields[key] = item
        else:
            raise KeyError(f"set key error: key {key} not in signature.  "
            "Cannot add new items.")
            
# -------------------------------------------------------------------------------------

class RaySet(SetBase):
    """
    All of the data that defines a set of rays, packaged.
    
    The most important feature provided by this tool is the ray set's signature, which
    is a set of strings that describes which types of data are being handled by this 
    set.  This will allow flexibility about which kinds of processing the ray tracer 
    executes.
    
    """
    
    def __init__(self, signature, sources, dimension, catagory="source", name="",
        include_default_signature=True, **kwargs
    ):
        """
        signature : set of strings
            A set of strings that name the data fields held by this ray set.  Does not
            need to include the default ray signature, x_start, y_start, z_start, 
            x_end, y_end, z_end, wavelength, as these will be added if parameter   
            add_default_signature is True (the default).  But if it is false, then
            you do need to explicitly name these default signature fields here.  Cannot
            be changed after instantiation.
        sources : List of sources, or single source
            Will typically be a source generated from one of the source classes in 
            sources.py, though may be anything that can be indexed with keys in the 
            signature.  Will attempt to hook source[sig] for sig in signature, and if 
            it fails for any reason, will hook none instead.  May be either a single 
            source, or a list of sources.  If it is a list, all sources in the list 
            will be concatenated together and used simultaneously.  Sources can be 
            shared between multiple ray sets.  Since the sources consumed by a set are 
            references, one ray set updating a source will cause other ray sets to be 
            changed.
        dimension : int
            Must be either 2 or 3, the number of spatial dimensions the rays reside in.
        catagory : string
            May be anything.  Describes what to do with this ray set.  "source", 
            "active", "dead", "finished", "stopped" are used by the default ray engine,
            but users might want to implement more types here.
        name : string
            May be anything, or left as an empty string.  An optional name for this ray
            set to help you keep track if you are using multiple sets together.
        include_default_signature : Bool
            If True, adds the default ray signature (see signature parameter) to the 
            signature set (so that self.signature = signature | default).  Otherwise
            the signature parameter is used unaltered.
            
        """
        if dimension not in {2, 3}:
            raise ValueError(
                f"RaySet: dimension must be 2 or 3, but was given {dimension}."
            )
        self._dimension = dimension
        self._catagory = catagory 
        self._name = name
        if type(signature) is not set:
            signature = set(signature)
        if include_default_signature:
            signature = signature | RaySet.generate_default_signature(dimension)
        else:
            signature = signature
            
        super().__init__(sources, signature, **kwargs)
            
    @classmethod
    def generate_default_signature(cls, dimension):
        if dimension == 2:
            return {"x_start", "y_start", "x_end", "y_end"}
        else:
            return {"x_start", "y_start", "z_start", "x_end", "y_end", "z_end"}
            
    @property
    def dimension(self):
        return self._dimension
        
    @property
    def catagory(self):
        return self._catagory
        
    @property
    def name(self):
        return self._name

# -------------------------------------------------------------------------------------
            
class BoundarySet(SetBase):
    """
    All of the data that defines a set of boundaries, packaged.
    
    The most important feature provided by this tool is the boundary set's signature, 
    which is a set of strings that describes which types of data are being handled by 
    this set.  This will allow flexibility about which kinds of processing the ray 
    tracer executes.
    
    """
    
    def __init__(self, signature, sources, catagory, name="", 
        include_default_signature=True, **kwargs
    ):
        """
        signature : set of strings
            A set of strings that name the data fields held by this boundary set.  
            Does not need to include the default boundary for this type of boundary, 
            as these will be added if parameter add_default_signature is True (the 
            default).  But if it is false, then you do need to explicitly name these 
            default signature fields here.  Cannot be changed after instantiation.
        sources :
            The source for the geometric data that defines the boundary.  Typically 
            this source will define the data fields in the default signature, though 
            it may define none, some, or all of the fields in the signature.
            
            Will attempt to hook source[sig] for sig in signature, and if it
            fails for any reason, will hook none instead.
        catagory : string
            Must be either "segment", "arc", or "triangle"
        name : string
            May be anything, or left as an empty string.  An optional name for this 
            boundary set to help you keep track if you are using multiple sets 
            together.
        include_default_signature : Bool
            If True, adds the default boundary signature for this catagory of boundary 
            (see the boundary set subclasses) to the signature set (so that 
            self.signature = signature | default).  Otherwise the signature parameter 
            is used unaltered.
            
        """
        
        self.name = name
        default_signature = self.process_catagory(catagory)
        if len(signature) == 0:
            signature = set()
        if include_default_signature:
            self.signature = signature | default_signature
        else:
            self.signature = signature
            
        super().__init__(sources, **kwargs)
                
    def process_catagory(self, catagory):
        signature = {
            "segment": {"x_start", "y_start", "x_end", "y_end"},
            "arc": {"x_center", "y_center", "angle_start", "angle_end", "radius"},
            "triangle": {"x0", "y0", "z0", "x1", "y1", "z1", "x2", "y2", "z2"}
        }
        try:
            self.catagory = catagory
            return signature[catagory]
        except KeyError:
            raise KeyError(f"BoundarySet: given invalid catagory {catagory}."
            "Must be 'segment', 'arc', or 'triangle'."
            )

# =====================================================================================

class RayEngine:
    """
    Builds and holds the tracing / optimization model.
    
    Here is how I envision this class working:
    Step 1:
        Call the RayEngine constructor, and feed it the dimension and a list of 
        RayOperations that define what kind of inputs and outputs you want from the
        engine.  This is a list because order matters, but it would prefer to be an
        ordered set, since we do not want repetitions.  Will check for exclusive 
        RayOperations and throw an error if it finds any.
        
        The RayEngine will contain five public attributes:
            1: input_signature, the signature of all RaySets input to the sytem.
            2: output_signature, the signature of all RaySets output from the system.
            3: optical_signature, the signature of all optically active BoundarySets.
            4: technical_signature, the signature of all non optically active 
                BoundarySets (stops and targets).
            5: material_signature, the signature of items in the materials list, if 
                used
    Step 2:
        Build each RaySet for the light inputs and each BoundarySet for the optical 
        elements.
    Step 3:
        Feed each set object into Ray_Engine.annotate to add the necessary annotations.
    Step 4:
        ...
        
    Every RayOperation that will ever be used by the script should be declared in the 
    constructor, and the set signatures should always posess every field that will 
    ever be used by the script.  But since the value of a field in a set may be none, 
    ray operations can be turned on and off as needed to speed things up when they are
    not needed.
        
    """
    
    def __init__(self, dimension, operations):
        if dimension not in {2, 3}:
            raise ValueError(
                f"RayEngine: dimension must be 2 or 3, but was given {dimension}."
            )
        self._dimension = dimension
        self._check_exclusions(operations)
        
    def _check_exclusions(self, operations):
        exclusions = set()
        used_operations = set()
        for op in operations:
            used_operations.add(op.__class__)
            exclusions = exclusions | op.exclusions
        exclusion_matches = used_operations & exclusions
        if bool(exclusion_matches):
            raise RuntimeError(
                f"RayEngine: discovered exclusive operations: {exclusion_matches}"
            )
        self.operations = operations
    
    
    
    
    
    
    
    
    
    
    
