"""
Optical operations that build upon the geometrical operations of the tracing engine.
"""

import tensorflow as tf

import tfrt.geometry as geometry

"""
Fields banned from being defined by operations, since they are already defined elsewhere:
x_start
y_start
x_end
y_end
x_center
y_center
angle_start
angle_end
radius
norm
catagory
valid
"""

class RayOperation:
    """
    Base class that holds a modular operation to be performed during ray tracing.
    
    Ray operations have two functions:
    1) They help the OpticalEngine determine what data signatures must be used at 
        different parts of the program.
    2) They can insert computations to be performed at various stages of ray tracing.
    
    Each ray operation can define a group of interdependant operations that need to be
    performed at different stages of ray tracing to implement a specific desired 
    feature.  There are several different times a ray operation can interact with data.  
    At least one of these functions should be overrided by client classes.
    1) Annotate: Only ever preformed manually.  Used before anything else, usually
        to add some field to a set.  Separated from preprocess because this may only
        need to be called rarely, like when the number of elements in a set changes.
        Annotate will be called by the engine, but works by adding fields to the sources
        contained in the optical system the engine is connected to.  
        This function takes one argument: the engine.
    2) Preprocess: Called after ray intersections have been calculated, but before any
        new rays have been created as a result of reactions.  This stage is for preparatory
        work that must be done before any ray has been created.
        This function takes two arguments: the engine, and the projection result.
    3) Main: Called after all preprocess operations have run.  This is the step where
        ray reactions will be generated.
        This function takes two arguments: the engine, and the projection result.
        It returns a dict that contains information about the generated rays.  This dict
        will have up some key that encodes how the ray was generated.  Valid options are:
        active, finished, stopped, dead, active_seg, finished_seg, stopped_seg, dead_seg, 
        active_arc, finished_arc, stopped_arc, dead_arc.  This can be used to enable ops to 
        perform geometry-dependant operations, although I don't see this being used often.  
        The value for each of these keys is another dictionary that contains two keys: 
        'rays', and 'valid'.  Rays contains the ray data for the generated rays, and valid 
        is True wherever the data in rays represents a real newly generated ray.  The data
        in rays should always map onto the projection result that created the ray, which
        is why we keep valid; sometimes we might not need to create a new ray from a 
        projection result, but we have to generate a dummy result so that postprocess can
        properly map between the new rays and the projection results.
    4) Postprocess: Performed after all ray reactions have been processed to perform 
        tweaks and adjustments to the generated rays.  This phase typically should not 
        create any new rays.
        This function takes two arguments: the engine, the projection result, and a dict 
        whose values are items created by Main and whose keys are the RayOperation that 
        created that item.
        
    Ray operations can be turned on and off.  The attribute active controls this.
    
    Exposes five properties: input_signature, output_signature,
    optical_signature, technical_signature, material_signature which should return a 
    set of strings (may be empty) that tells the RayEngine what signature fields 
    are used by this operation.  Defaults to empty, so a subclass needs to override 
    these if it will use a signature.
    
    input_signature : signature of sources (rays input to the system).
    output_signature : signature of rays output from the system (generated by reactions)
    optical_signature : signature of optically active surfaces
    stop_signature : signature of stop surfaces.
    target_signature : signature of target surfaces.
    material_signature : signature of materials.
    
    #The fields in each of these locations will be easily accessible to operations EXCEPT
    #for the fields in input_signature, which are intended to be used for other purposes,
    #like optimization.
    
    #Exposes a property: simple_ray_inheritance, a set of signatures that MUST be present
    #in both input_signature and output_signature.  These fields will be automatically
    #copied from the parent ray as new rays are generated by the engine, without any extra
    #handling logic required by the operation.  If your field needs more than a simple copy,
    #don't include it here, and the operation will need to its own code to process it.
    
    Exposes four methods: annotate, preprocess, main, postprocess.  By default 
    these do nothing, but any subclass should always override at least one of these.
    Should be overridden with a function that will be called during tracing to 
    actually accomplish the work done by this class.  These functions should accept
    one argument: ray_engine.
    
    Exposes an attribute: exclusions, which is a set of other RayOperations that 
    this RayOperation is exclusive with.  Checked by the RayEngine and will throw an 
    error if exclusive RayOperations are specified. 
    
    """
    def __init__(self, active=True):
        self.active = active
        """
        # assert that simple_ray_inheritance is implemented correctly
        assert (self.simple_ray_inheritance <= self.input_signature),\
            f"Operation {type(self)}: Simple ray inheritance "\
            f"{self.simple_ray_inheritance} must be a subset of input "\
            f"signature {self.input_signature}"
        assert (self.simple_ray_inheritance <= self.output_signature),\
            f"Operation {type(self)}: Simple ray inheritance "\
            f"{self.simple_ray_inheritance} must be a subset of output "\
            f"signature {self.output_signature}"
        """
    
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
    def stop_signature(self):
        return set()
        
    @property   
    def target_signature(self):
        return set()
    
    @property
    def material_signature(self):
        return set()
        
    @property
    def simple_ray_inheritance(self):
        return set()
       
    def annotate(self, engine):
        pass
        
    def preprocess(self, engine, proj_result):
        pass
    
    def main(self, engine, proj_result):
        return {}
    
    def postprocess(self, engine, proj_result, new_rays):
        pass
            
    @property
    def exclusions(self):
        return set()

# ======================================================================================

class OldestAncestor(RayOperation):
    """
    Allows output rays to be related to the input ray from which they originate.
    
    Needs annotate to be called manually, at least once on startup to populate the sources'
    fields, and also whenever the number of source rays changes.
    """
       
    @property
    def input_signature(self):
        return {"oldest_ancestor"}
    
    @property
    def output_signature(self):
        return self.input_signature
        
    @property
    def simple_ray_inheritance(self):
        return self.input_signature
        
    def annotate(self, engine):
        start = 0
        system = engine.optical_system
        for source in system._sources:
            item_count = source["x_start"].shape[0]
            end = start + item_count
            source["oldest_ancestor"] = tf.range(start, end)
            start = end
            
    def postprocess(self, engine, proj_result, new_rays):
        pass
        
# -------------------------------------------------------------------------------------

class StandardReaction(RayOperation):
    """
    Generates output rays with refraction and possibly reflection.
    """
    def __init__(self, refractive_index_type="index", **kwargs):
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
        super().__init__(**kwargs)
        if refractive_index_type in {"index", "value"}:
            self._refractive_index_type = refractive_index_type
        else:
            raise ValueError(f"StandardReaction: received invalid value "
                "{refractive_index_type}.  Must be 'index' or 'value'."
            )
        
    @property
    def input_signature(self):
        if self._refractive_index_type == "index":
            return {"wavelength"}
        else:
            return set()
    
    @property
    def output_signature(self):
        return self.input_signature
        
    @property
    def simple_ray_inheritance(self):
        return self.input_signature
   
    @property
    def optical_signature(self):
        if self._refractive_index_type == "index":
            return {"mat_in", "mat_out"}
        else:
            return {"n_in", "n_out"}

    @property
    def material_signature(self):
        if self._refractive_index_type == "index":
            return {"n"}
        else:
            return set()
        
    def main(self, engine, proj_result):
        if "active" not in proj_result["rays"].keys():
            return {}
        
        rays = proj_result["rays"]["active"]
        
        if self._refractive_index_type == "index":
            mat_in = proj_result["optical"]["mat_in"]
            mat_out = proj_result["optical"]["mat_out"]
            wavelength = rays["wavelength"]
            
            n_stack = tf.stack(
                [mat["n"](wavelength) for mat in engine.optical_system.materials]
            )
            ray_range = tf.range(tf.shape(rays["x_start"])[0], dtype=tf.int64)
            n_in_indices = tf.stack([mat_in, ray_range], axis=1)
            n_out_indices = tf.stack([mat_out, ray_range], axis=1)
            n_in = tf.gather_nd(n_stack, n_in_indices)
            n_out = tf.gather_nd(n_stack, n_out_indices)
        else:
            n_in = proj_result["optical"]["n_in"]
            n_out = proj_result["optical"]["n_out"]
        
        new_rays = {}
        new_rays["x_start"], new_rays["y_start"], new_rays["x_end"], new_rays["y_end"] =\
            geometry.snells_law(
                rays["x_start"],
                rays["y_start"],
                rays["x_end"], 
                rays["y_end"],
                proj_result["optical"]["norm"],
                n_in,
                n_out,
                engine.new_ray_length
            )
        valid = tf.broadcast_to(True, tf.shape(rays["x_end"]))
        return {"active": {"rays": new_rays, "valid": valid}}
        
# -------------------------------------------------------------------------------------

class GhostThrough(RayOperation):
    """
    Simple test case, rays just pass through optical surfaces.
    """   
    
    def main(self, engine, proj_result):
        rays = proj_result["rays"]["active"]
        new_rays = {
            "x_start": rays["x_end"],
            "y_start": rays["y_end"],
            "x_end": 2 * rays["x_end"] - rays["x_start"],
            "y_end": 2 * rays["y_end"] - rays["y_start"]
        }
        valid = tf.broadcast_to(True, tf.shape(rays["x_end"]))
        return {"active": {"rays": new_rays, "valid": valid}}

    
    
    
    
    
    
    
    
    
    
    