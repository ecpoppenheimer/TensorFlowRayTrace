"""
Optical operations that build upon the geometrical operations of the tracing engine.
"""

from abc import ABC, abstractmethod

class RayOperation(ABC):
    """
    Base class that holds a modular operation to be performed during ray tracing.
    
    Ray operations have two functions:
    1) They help the RayEngine determine what data signatures must be used at 
        different parts of the program.
    2) They can insert computations to be performed at various stages of ray tracing.
    
    Each ray operation can define a group of interdependant operations that need to be
    performed at different stages of ray tracing to implement a specific desired 
    feature.  There are four different times a ray operation can interact with data.  
    At least one should be used.
    1) Annotation: Only ever performed manually by the user.  Essentially an optional
        first step that can be called whenever the inputs change in a way that requires
        them to be re-labeled, such as changing the number of rays being traced.
    2) Preprocessing: Performed during a ray tracing step, but before any of the
        intersection computation is performed, usually to write various 
        annotations to a data set.  Useful if the annotations need to always be
        recomputed before tracing.
    3) Main: Called after ray intersections have been called.  This is the step where
        ray reactions will be generated.
    4) Postprocessing: Performed after all ray reactions have been processed.  At this
        point all rays that will be created should have been created.
        
    Ray operations can be turned on and off.  The attribute active controls this.
    
    Exposes five properties: input_signature, output_signature,
    optical_signature, technical_signature, material_signature which should return a 
    set of strings (may be empty) that tells the RayEngine what set signature fields 
    are used by this operation.  Defaults to empty, so a subclass needs to override 
    these if it will use a signature.
    
    Exposes four attributes: preprocess, main, postprocess, annotate.  By default 
    these are None, but any subclass should always override at least one of these.
    Should be overridden with a function that will be called during tracing to 
    actually accomplish the work done by this class.
    
    Exposes an attribute: exclusive_with, which is a set of other RayOperations that 
    this RayOperation is exclusive with.  Checked by the RayEngine and will throw an 
    error if exclusive RayOperations are specified. 
    
    """
    def __init__(self, active=True):
        self.active = active
    
    @property   
    def input_signature(self):
        return set()
    
    @property   
    def output_signature(self):
        return set()
    
    @property   
    def optical_signature(self):
        return set()
    
    @property   
    def technical_signature(self):
        return set()
    
    @property
    def material_signature(self):
        return set()
        
    @property
    def preprocess(self):
        return None
    
    @property
    def main(self):
        return None
    
    @property
    def postprocess(self):
        return None
            
    @property
    def annotate(self):
        return None
            
    @property
    def exclusions(self):
        return set()

# -------------------------------------------------------------------------------------

class OldestAncestor(RayOperation):
    """
    Allows output rays to be related to the input ray from which they originate.
    """
       
    @property
    def input_signature(self):
        return {"oldest_ancestor"}
    
    @property
    def output_signature(self):
        return {"oldest_ancestor"}
        
    def annotate(self, ray_engine):
        ray_set["oldest_ancestor"] = tf.range(tf.shape(ray_set["x_start"])[0])
        
# -------------------------------------------------------------------------------------

class StandardReaction(RayOperation):
    """
    Generates output rays with refraction and possibly reflection.
    """
    def __init__(self, active, refractive_index_type="index"):
        """
        refractive_index_type : string
            Must be either "index" or "value".  
            If "value", then requires each optical boundary to have two fields: "n_in" 
            and "n_out" which are the refractive index inside and outside of the 
            material.  They should have type tf.float64.
            If "index", then requires each optical boundary to have two fields: 
            "n_in_index" and "n_out_index" which are indices that reference a material
            from the materials list given to the RayEngine.  Should have type tf.int64.
            Also requires all rays to have the field "wavelength", a tf.float64 that
            describes the wavelength of the light in nm.  These will be used to compute
            the refractive index used to perform the ray reaction.
        """
        super().__init__(active)
        if refractive_index_type in {"index", "value"}:
            self._refractive_index_type = refractive_index_type
        else:
            raise ValueError(f"StandardReaction: received invalid value "
                "{refractive_index_type}.  Must be 'index' or 'value'."
            )
        
    @property
    def input_signature(self):
        return {"wavelength"}
    
    @property
    def output_signature(self):
        return {"wavelength"}
   
    @property
    def optical_signature(self):
        if self._refractive_index_type == "index":
            return {"n_in_index", "n_out_index"}
        else:
            return {"n_in", "n_out"}
   
    @property
    def material_signature(self):
        if self._refractive_index_type == "index":
            return {"n_in", "n_out"}
        else:
            return None

    
    
    
    
    
    
    
    
    
    
    
