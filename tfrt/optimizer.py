"""
Classes that help set up the optimization process.
"""

import tensorflow as tf
import time

class SGD_Optimizer:
    """
    Class that defines and executes the optimization process.
    
    This class is built around TF's SGD optimizer, but includes processing and helper 
    functionality tuned toward TFRT's needs.
    
    Most users will want to optimize their system by either calling single_step inside a
    loop, or by calling training_routine.
    
    There are two kinds of matricies that this class can use to help stabilize the optimization
    process, which bear explanation: accumulators and smoothers.
    
    An accumulator is a 2x2 square matrix with side length equal to the number of 
    parameters in an element.  It will be left matrix multiplied with the gradient and
    has the effect of performing a cumulative sum, which greatly helps streamline the
    optimization process.  For a 2D optical surface, the accumulator will generally be
    a triangular matrix, with ones along and to one side of the main diagonal and zeros 
    elsewhere.  For a 3D optical surface, things are more complicated, but it can be 
    calculated by tfrt.mesh_tools.mesh_parametrization_tools.
    
    A smoother is a matrix that can be left matrix multiplied with the parameters to
    smooth out the surface, in a way analogous to Gaussian smoothing.  These matrices can
    be generated by tfrt.mesh_tools.mesh_smoothing_tool.  A smoothing matrix must be square 
    with side length equal to the flattened size of parameters.
    
    Valid formats for feeding either an accumulator or smoother to this class include None, a 
    single one, or a list of them and/or None.  If the input is a single element, that element 
    is used for every parameter, and if the input is a list, it must have one element per 
    parameter.  None values will prevent the use of that kind of matrix for that surface.
    
    Parameters
    ----------
    engine : a tfrt.engine.OpticalEngine
        The optical engine that controls the ray tracing.
    parameters : a list of tf.Variable
        The tf.Variables whose values will be adjusted by the optimizer to solve the problem.
    error_function : function
        This is a function that will calculate the error that will be used to generate the
        gradient.  The function must take one required arguments: engine.  It may take
        additional args or kwargs, which can be passed to the step or routine functions.
        Warning that the following identifiers could cause conflicts:
        args: accumulators
        kwargs: lr_scale, momentum, verbose
    trace_depth : int
        The number of ray tracing steps to perform while evaluating the error.
    
    Optional Parameters
    -------------------
    momentum : float
        The momentum hyperparameter for the optimizer.  Must be between 0 and 1.
    learning_rate : float
        The learning rate hyperparameter for the optimizer.
    individual_lr : None or list/tuple of floats.
        If not None, must contain as many elements as are in parameters, in which case each 
        element is a modifier for the learning rate for that set of parameters.  The value
        is multiplied with learning_rate, to determine the actual learning rate that will
        be used by each individual optic, which enables different learning rates for different 
        optics.  And optimization for an optic element can even be turned off by setting its 
        value to zero.
    grad_clip : float or None
        If not None, all gradients are clipped to lie within +/- this value.  Prevents
        extreme changes being made to the parameters.  Defaults to 10x the learning rate
    clip_mode : str, optional
        Defaults to 'common', in which case the same clip value is used for all grads.
        May also be 'individual', in which case the clip value if proportional to the 
        individual learning rates.
    clip_scale : float, optional
        Defaults to 10.  The default size of the clip value, relative to the learning rate.
        
    Public read/write attributes
    ----------------------------
    All parameters and optional parameters are also read/write attributes.
    
    Public methods
    --------------
    convert_to_plist
    process_gradient
    smooth
    single_step
    training_routine
    """
    def __init__(
        self,
        engine, 
        parameters,
        error_function,
        trace_depth, 
        momentum=0.0,
        learning_rate=1.0,
        individual_lr=None,
        grad_clip="default",
        clip_mode="common",
        clip_scale=10.0
    ):
        self._opt = tf.optimizers.SGD(nesterov=True)
        
        self.engine = engine
        if type(parameters) is list or type(parameters) is tuple:
            self.parameters = parameters
        else:
            raise ValueError("SGD_Optimizer: parameters must be a list of tf.variable")
        self.error_function = error_function
        self.trace_depth = trace_depth
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.individual_lr = individual_lr
        self.clip_scale = clip_scale
        if grad_clip == "default":
            self.grad_clip = self.clip_scale * learning_rate
        else:
            self.grad_clip = grad_clip
        self.clip_mode = clip_mode
        self.suppress_warnings = False
        self.iterations = 0
    
    @property
    def momentum(self):
        return self._momentum
        
    @momentum.setter
    def momentum(self, val):
        if val <= 1.0 and val >= 0.0:
            self._opt.momentum = val
            self._momentum = val
        else:
            raise ValueError(
                "SGD_Optimizer: Momentum must be between 0 and 1."
            )
        
    @property
    def individual_lr(self):
        return self._individual_lr
        
    @individual_lr.setter
    def individual_lr(self, val):
        if val is None:
            self._individual_lr = [1.0] * len(self.parameters)
        else:
            try:
                if len(val) != len(self.parameters):
                    raise ValueError(
                        "SGD_Optimizer: individual_lr must have as many elements as there are "
                        "parameters."
                    )
                else:
                    self._individual_lr = val
            except(TypeError) as e:
                raise TypeError(
                    "SGD_Optimizer: individual_lr must have as many elements as there are "
                    "parameters."
                ) from e

    def convert_to_plist(self, data):
        """
        Check/convert the input to a list with as many elements as parameters.
        
        training_routine will automatically use this function to convert accumulator and
        smoother arguments to the correct format.
        """
        p_count = len(self.parameters)
        
        if type(data) is list or type(data) is tuple:
            if len(data) == p_count:
                return data
            else:
                raise ValueError(
                    "SGD_Optimizer: plist arguments must have one element per "
                    "parameter."
                )
        else:
            return [data] * p_count
            
    def convert_to_lrlist(self, lr, steps):
        try:
            return tf.linspace(lr[0], lr[1], steps).numpy()
        except(TypeError):
            return [lr] * steps
                
    def process_gradient(self, accumulators, *args, lr_scale=1.0, **kwargs):
        """
        Compute and process the gradients.
        
        Though this function is public, since calling it shouldn't cause any problems, it
        also generally won't be needed by most users.
        
        Parameters
        ----------
        accumulators : list of accumulator matricies
            See the class documentation for a description of what goes here.
        lr_scale : float, optional
            This value is multiplied with learning_rate to determine the actual learning
            rate that will be used to adjust the parameters.  This is a convenience so that
            you can enter the learning rate once, and easily adjust its relative value during
            optimization.  Basically, change this value as you optimize, rather than changing
            learning_rate.
            
        Additional args and kwargs can be given to this function.  They will be passed to the
        error function.
        
        Returns
        -------
        grads : 
            A list containing the updates that should be made to the parameters.
        error :
            A single float that is the summed value of the error used for this step.
        """
        self.engine.clear_ray_history()
        with tf.GradientTape() as tape:
            self.engine.optical_system.update()
            self.engine.ray_trace(self.trace_depth)
            error = self.error_function(self.engine, *args, **kwargs)
            grads = tape.gradient(error, self.parameters)
        
        processed_grads = []    
        for i in range(len(self.parameters)):
            grad = grads[i]
            params = self.parameters[i]
            
            # ensure the gradient exists and is finite.
            try:
                grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
            except(ValueError):
                if self.suppress_warnings:
                    print(
                        "Warning: SGD_Optimizer.process_gradient encountered a possible " 
                        "issue:  The gradient was likely None, which can mean that the error "
                        "does not depend on it.  This may or may not mean that things aren't "
                        "working.  The gradient will be set to zero and future instances of "
                        "this message will be suppressed."
                    )
                    self.suppress_warnings = True
                grad = tf.zeros_like(params, dtype=tf.float64)
                
            # apply the learning rate and clip.
            grad *= lr_scale * self.individual_lr[i] * self.learning_rate
            if self.clip_mode == "common":
                grad = tf.clip_by_value(grad, -self.grad_clip, self.grad_clip)
            else:
                clp = self.individual_lr[i] * self.clip_scale * self.learning_rate * lr_scale
                grad = tf.clip_by_value(grad, -clp, clp)
            
            # apply the accumulator
            if accumulators[i] is not None:
                prev_shape = grad.shape
                grad = tf.reshape(grad, (-1, 1))
                grad = tf.matmul(accumulators[i], grad)
                grad = tf.reshape(grad, prev_shape)
               
            processed_grads.append(grad)
        return processed_grads, tf.reduce_mean(error)
    
    @staticmethod    
    def smooth(parameters, smoother):
        """
        Smooth a surface via its parameters.
        
        This method takes in a single set of parameters, and a smoothing matrix which can be
        generated by tfrt.mesh_tools.mesh_smoothing_tool, and smooths the surface of the
        optic.  The value of the parameters is updated, but the surface itself is not.
        
        Parameters
        ----------
        parameters : tf.variable
            The parameters of the surface to smooth
        smoother : rank 2 float tensor
            The smoothing matrix that will be applied to the parameters.  Must be square with
            side length equal to the flattened size of parameters.
        """
        if smoother is not None:
            prev_shape = parameters.shape
            params = tf.reshape(parameters, (-1, 1))
            params = tf.matmul(smoother, params)
            params = tf.reshape(params, prev_shape)
            parameters.assign(params)
        
    def single_step(
        self, accumulators, *args, lr_scale=1.0, momentum=0.0, verbose=False, **kwargs
    ):
        """
        Run a single optimization step.
        
        Parameters
        ----------
        accumulators : list of accumulator matricies
            See the staticmethod check_accumulators for a description of what goes here.
        lr_scale : float, optional
            The relative learning rate.  Multiplied with learning_rate to determine the actual
            step size.
        momentum : float, optional
            Defaults to zero.  The momentum hyperparameter, which is effectively the fraction
            of the previous gradient step to add to the current one.  Must be between 0 and 1,
            where zero means no momentum.  Higher values can speed up slow learning but also
            cause the solution to oscillate around the ideal value.
        verbose : bool, optional
            Defaults to False, in which case no message is displayed.  If True, will print
            a message displaying the current iteration number and the error.
            
        Additional args and kwargs can be given to this function which will be passed to the
        error function.
        
        Returns
        -------
        error : float
            The total summed error from this step.
        """
        self.momentum = momentum
        grads, error = self.process_gradient(accumulators, *args, lr_scale=lr_scale, **kwargs)
        self._opt.apply_gradients([(g, p) for g, p in zip(grads, self.parameters)])
        self.iterations += 1
        if verbose:
            print(f"step {self.iterations} error: {error.numpy()}")
        return error.numpy()
            
    def training_routine(
        self, 
        routine, 
        post_step=None,
        report_frequency=1, 
        show_time=True
    ):
        """
        Run many optimization steps defined in a routine.
        
        The routine must be a list of dictionaries.  Each dictionary defines hyperparameters
        for a phase of optimization steps.  When this function is called, it will generate
        a default phase dictionary with the values:
        {
            "steps": 10,
            "learning_rate": 1.0,
            "momentum": 0.0,
            "accumulators": None,
            "smoothers": None,
            "erf_args": [],
            "erf_kwargs": {},
            "individual_lr": None
        }
        Each optimization phase is defined by a similar dictionary.  Each time a new phase 
        begins, the phase dictionary will be updated with the values specified in the next 
        element of the routine.  Thus you do not need to define hyperparameters if you are 
        happy with the default, or if they do not need to change from the values of the 
        previous phase.
        
        Please note that 'learning_rate' in this definition is actually the relative learning
        rate, which is called lr_scale in single_step and process_gradient.  It's value should
        generally be between 0 and 1, and is used to perform a relative reduction of the
        learning rate hyperparameter as training procedes.
        
        'learning_rate' may be a float, or it may be a tuple of two floats, in which case it
        will be continuously varied changed from the first to the second throughout the
        phase.
        
        Parameters
        ----------
        routine : list of dicts
            See the above description.
        report_frequency : int, optional
            If zero, will print nothing.  If nonzero, will periodically print a message 
            displaying the progress after this many steps have been taken.
        show_time : bool, optional
            If true will print a message displaying the time to run the optimization and
            average step time.
        post_step : callable, optional
            A function to call after each training step.  This is generally used for something
            like redrawing the display.
        """
        phase = {
            "steps": 10,
            "learning_rate": 1.0,
            "momentum": 0.0,
            "accumulators": None,
            "smoothers": None,
            "erf_args": [],
            "erf_kwargs": {},
            "individual_lr": None
        }
        self.iterations = 0
        phase_count = len(routine)
        total_iterations = 0
        steps = phase["steps"]
        start_time = time.time()
        
        # count total_iterations
        for new_phase in routine: 
            try:
                steps = new_phase["steps"]
            except(KeyError):
                pass
            total_iterations += steps
            
        # actually run the phases
        current_phase = 0
        for new_phase in routine:
            current_phase += 1
            phase_iterations = 0
            phase.update(new_phase)
            phase["accumulators"] = self.convert_to_plist(phase["accumulators"])
            phase["smoothers"] = self.convert_to_plist(phase["smoothers"])
            phase["learning_rate"] = self.convert_to_lrlist(
                phase["learning_rate"],
                phase["steps"]
            )
            self.individual_lr = phase["individual_lr"]
            for i in range(phase["steps"]):
                error = self.single_step(
                    phase["accumulators"],
                    *phase["erf_args"],
                    lr_scale=phase["learning_rate"][i],
                    momentum=phase["momentum"],
                    verbose=False,
                    **phase["erf_kwargs"]
                )
                
                # smooth everything None smoothers will have no effect
                for p, s in zip(self.parameters, phase["smoothers"]):
                    self.smooth(p, s)
                phase_iterations += 1
                
                if report_frequency != 0:
                    if self.iterations % report_frequency == 0:
                        print(
                            f"Phase {current_phase}/{phase_count}, "
                            f"step {phase_iterations}/{phase['steps']}, "
                            f"total {self.iterations}/{total_iterations}-"
                            f"{100*self.iterations/total_iterations:.1f}%.  "
                            f"Error: {error}."
                        )
                        
                if post_step:
                    post_step()
                        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Completed training routine.  Took {total_time} seconds.")
        print(f"Steps took an average of {total_time/total_iterations} seconds per step.")
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
