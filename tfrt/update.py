from abc import ABC, abstractmethod

class RecursivelyUpdatable(ABC):
    """
    Base class for objects that are recursively updatable.
    
    Has a public method, update, that takes no arguments, and causes all of this
    classes internal data to be recomputed.  Designed to be used with TensorFlow eager
    execution, and in this case will cause tensor values to be recalculated.
    This object is able to recursively cause sub objects to update as well, and is
    designed to facilitate that behavior.
    
    Has a public attribute: recursively_update, which can be changed on the fly, which
    will can turn recursive updating on and off.  False -> no recursive update.
    
    Has a public attribute: update_handles, which is a list of update functions that
    will be called before calling self.update.  These functions must take no arguments.
    They will only be called if recursively_update is True.
    
    Classes that inherit from this base class should call super().__init__(**kwargs)
    late in their constructor, and should be able to pass update_handles and 
    recursively_update to it.  Classes that inherit from this must implemented _update,
    a function that takes no arguments and does the work of updating the class itself.
    This base class will do the work of updating the subclasses.  Inheriting classes
    must also implement _generate_update_handles(), which takes no arguments and
    generates a list of update handles for the sub-classes that this object will 
    recursively update.
    """
    
    def __init__(self, update_handles=None, recursively_update=True):
        self.recursively_update = recursively_update
        if update_handles is None:
            self.update_handles = self._generate_update_handles()
        else:
            self.update_handles = update_handles
        self.update()
        
    def update(self):
        if self.recursively_update:
            for handle in self.update_handles:
                handle()
        self._update()
        
    @abstractmethod
    def _update(self):
        raise NotImplementedError
        
    @abstractmethod
    def _generate_update_handles(self):
        """
        fill self.update_handles with a list of update functions for its consumed 
        distributions.
        """
        raise NotImplementedError
