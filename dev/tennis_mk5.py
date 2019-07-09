import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.transform import Rotation

import tfrt.sources as sources
import tfrt.TFRayTrace as ray_tracer
import tfrt.drawing as drawing
import tfrt.materials as materials
import tfrt.OpticsUtilities as outl

import context_manager as cm

def calculate_tetrahedron(h, l, b, theta):
    """
    h : float
        The vertical height from the optic center to the focal plane.
    l : float
        The horizontal distance between the optical center (projected onto the 
        perpendicular focal plane) and the location where the beam center 
        strikes the focal plane.  l/h is the tangent of the angle at which the 
        source plane is tilted, relative to vertical.
    b : float
        The horizontal distance in the focal plane between where the beam 
        center strikes it and the edge of the light distribution.  
        Characterizes the spread of the beam at theta = 0.
    theta : float
        The angle at which we are calculating the cross section, relative to 
        the inclination of the beam center.
        
    Returns
    length : float
        The length of the cross section that the distribution should cover.
    plane_angle : float
        The angle theta, but projected into the focal plane.  The angle 
        between the beam cross section intersection with the focal plane and 
        theta = 0.
    beam_angle : float
        The angle between the end of the beam and the source, relative to
        the beam center.
    
    """
    theta = np.radians(theta)
    cos_theta = np.cos(theta)**2
    h2 = h**2
    l2 = l**2
    b2 = b**2
    
    length = np.sqrt((b2*h2 + b2*l2*cos_theta)/(h2*cos_theta + l2*cos_theta))
    plane_angle = np.degrees(np.arccos(b / length))
    beam_angle = np.degrees(np.arcsin(np.sqrt(
        1 - l2*b2/length**2/(h2+l2)
    )))
    
    return length, plane_angle, beam_angle

def calculate_section_boundaries(
    height,
    horizontal_offset,
    beam_width_forward,
    beam_width_sideways_forward,
    angle,
    beam_width_backward=None,
    beam_width_sideways_backward=None
):
    """
    height : float
        The vertical distance between the optic center and the focal plane.
    horizontal_offset : float
        The horizontal distance in the focal plane between the point directly 
        below the optic center and where the center of the beam strikes the 
        focal plane.  Encodes the angle at which the optic is tilted relative 
        to the focal plane. Will be zero if the beam center is vertical.
    beam_width_forward : float
        The horizontal distance in the focal plane between where the beam 
        strikes it and the edge of the light distribution, measured in the 
        same direction as the tilt (optic is tilted forward).  Note that the 
        full beam width will actually be beam_width_forward + 
        beam_width_backward; this value is more akin to radius than
        diameter.
    beam_width_sideways_forward : float
        Like beam_width_forward, but in the direction perpendicular to it; the 
        half-width of the light distribution in the other dimension.  This 
        dimension will be used to cut off the forward part of the distribution 
        to make one side of the square light distribution. 
    angle : float
        May be a single value or a numpy array of values.  The angles at which 
        cross sections of the optic are being taken.
    beam_width_backward : optional, float
        Same as beam_width_forward but for the opposite side.  Defaults to the 
        value of beam_width_forward.  Used only if the beam is not symmetric 
        about its center in this dimension.
    beam_width_sideways_backward : optional, float
        Same as beam_width_sideways_forward, but for the opposite side.  
        Defaults to the value of beam_width_sideways_forward.  Used only if 
        the beam is not symmetric about its center in this dimension.
        
    Returns:
    forward_section : float
        The half-width of the beam cross section, projected into the focal 
        plane, for a cross section at each angle.  Will be a np.array if more 
        than one angle was given.
    backward_section : float
        Same as above, but the back side of the beam cross section.  Or, is the
        half-width of the beam cross section at each angle + PI.
    beam_angle : float
        The angle between the end of the beam and the source, relative to
        the beam center.
    slice_center_offset : float
        The horizontal distance between the optic center and where the beam center
        strikes the focal plane, projected into the slice plane
    
    """
    
    beam_width_backward = beam_width_backward or beam_width_forward
    beam_width_sideways_backward = (
        beam_width_sideways_backward or 
        beam_width_sideways_forward
    )
        
    forward_section, plane_angle, beam_angle = calculate_tetrahedron(
        height,
        horizontal_offset,
        beam_width_forward,
        angle
    )
    backward_section, _, _ = calculate_tetrahedron(
        height,
        horizontal_offset,
        beam_width_backward,
        angle
    )
    
    # cut off the distribution to form the sides
    sin_plane_angle = np.sin(np.radians(plane_angle))
    forward_side_length = forward_section * sin_plane_angle
    backward_side_length = backward_section * sin_plane_angle
    
    denominator = np.where(sin_plane_angle > 0.0, sin_plane_angle, 1.0)
    
    forward_section = np.where(
        np.less(forward_side_length, beam_width_sideways_forward),
        forward_section,
        beam_width_sideways_forward / denominator
    )
    backward_section = np.where(
        np.less(backward_side_length, beam_width_sideways_backward),
        backward_section,
        beam_width_sideways_backward / denominator
    )
    
    slice_center_offset = (
        np.sqrt(height**2 + horizontal_offset**2) * np.cos(np.radians(beam_angle))
    )
    
    return forward_section, backward_section, beam_angle, slice_center_offset
           
# =============================================================================
 
# Major parameters
# distances are in inches, angles in degrees
height = 168.0
horizontal_offset = 132.0
beam_width_forward = 180.0
beam_width_sideways_forward = 228.0
slice_angles = np.linspace(0.0, 90.0, 10)

ray_count = 200
source_angular_cutoff = 75
source_aperature_radius = 0.059
source_diameter_sample_count = 5

lens_segment_count = 64 # Must be even or else the middle will not be fixed.
lens_angular_cutoff = 90
lens_material = materials.acrylic

surface_polynomial_order = 16

max_target_size = 10000.0
gradient_clip = 0.1

VISUALIZE = True
VISUALIZE_UPDATE_FREQUENCY = 1
PRINT = False

# Calculated parameters
source_angular_cutoff = np.radians(source_angular_cutoff)
lens_angular_cutoff = np.radians(lens_angular_cutoff)

# build all the placeholders here, so they can be stored in the context manager
lens_first_fixed_point = tf.placeholder(tf.float64, (), "lens_first_fixed_point")
first_clip_minus = tf.placeholder(tf.float64, (), "first_clip_minus")
first_clip_plus = tf.placeholder(tf.float64, (), "first_clip_plus")
lens_second_fixed_point = tf.placeholder(tf.float64, (), "lens_second_fixed_point")
second_clip_minus = tf.placeholder(tf.float64, (), "second_clip_minus")
second_clip_plus = tf.placeholder(tf.float64, (), "second_clip_plus")

target_definition = tf.placeholder(tf.float64, (2, 3), "target_definition")
target_selector = tf.placeholder(tf.int64, (), "target_selector")

segment_selector = tf.placeholder(tf.string, (), "segment_selector")

central_angle = tf.placeholder(tf.float64, (), "central_angle")
source_selector = tf.placeholder(tf.string, (), "source_selector")

momentum = tf.placeholder(tf.float64, (), "momentum")
first_LR = tf.placeholder(tf.float64, (), "first_LR")
second_LR = tf.placeholder(tf.float64, (), "second_LR")

placeholder_list = [
    lens_first_fixed_point,
    first_clip_minus,
    first_clip_plus,
    lens_second_fixed_point,
    second_clip_minus,
    second_clip_plus,
    target_definition,
    target_selector,
    segment_selector,
    central_angle,
    source_selector
]

def build_context():
    context = cm.ContextManager(True, True)
    context.default_feeds.update({
        lens_first_fixed_point: .1,
        first_clip_minus: .025,
        first_clip_plus: .3,
        lens_second_fixed_point: .35,
        second_clip_minus: .2,
        second_clip_plus: .6
    })

    # perform the beam angle trig
    forward_section, backward_section, bas, slice_center_offset = \
        calculate_section_boundaries(
            height,
            horizontal_offset,
            beam_width_forward,
            beam_width_sideways_forward,
            slice_angles
        )
    beam_angles = {sl: -np.radians(bm) for sl, bm in zip(slice_angles, bas)}
    
    # build the ray target definition for the full lens
    second_target_definition_dict = {
        slice_angle: np.array([
            offset - backward,
            offset + forward,
            1.0
        ], dtype=np.float64) 
        for slice_angle, forward, backward, offset in zip(
            slice_angles,
            forward_section,
            backward_section,
            slice_center_offset
        )
    }

    # build the ray targets for just the first lens.  Try to get it to do half of the 
    # bending work.
    first_target_definition_dict = {}
    for slice_angle in slice_angles:
        sec_target = second_target_definition_dict[slice_angle]
        beam_angle = beam_angles[slice_angle]
        
        back_angle = (
            np.arctan2(-height, sec_target[0]) + beam_angle - source_angular_cutoff
        ) / 2.0
        fore_angle = (
            np.arctan2(-height, sec_target[1]) + beam_angle + source_angular_cutoff
        ) / 2.0
        
        back_edge = -height / np.tan(back_angle)
        fore_edge = -height / np.tan(fore_angle)
        first_target_definition_dict[slice_angle] = np.array(
            [back_edge, fore_edge, 1.0]
        )
        
    # build the slice angle context
    context.add_context(
        "slice_angle",
        {
            slice_angle: (
                {
                    central_angle: beam_angles[slice_angle],
                    target_definition: np.stack([
                        first_target_definition_dict[slice_angle],
                        second_target_definition_dict[slice_angle]
                    ])
                },
                {}
            ) for slice_angle in slice_angles
        },
        slice_angles[0]
    )
    
    # build the active surface context
    context.add_context(
        "surface",
        {
            "first": (
                {
                    target_selector: 0,
                    segment_selector: "first"
                },
                {}
            ),
            "both": (
                {
                    target_selector: 1,
                    segment_selector: "both"
                },
                {}
            )
        },
        "both"
    )
    
    # build the active source context
    context.add_context(
        "source",
        {
            "training": (
                {
                    source_selector: "training"
                },
                {}
            ),
            "printing": (
                {
                    source_selector: "printing"
                },
                {}
            ),
            "inspection": (
                {
                    source_selector: "inspection"
                },
                {}
            )
        },
        "training"
    )
    return context

# Either create the context manager by loading from a file (and so keeping a previous
# solution / program state) or generate a new context.
context_file_name = "tennis_context.dat"
if True: # load a previous context
    try:
        context = cm.load(context_file_name, placeholder_list)
    except FileNotFoundError:
        print("WARNING: Could not open the context file, so building new context"
            "instead"
        )
        context = build_context()
else: # generate a new context
    context = build_context()

# set up the drawing stuff
drawing.disable_figure_key_commands()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
if VISUALIZE:
    plt.show(block=False)

ax.set_aspect("equal")
#ax.set_xbound(-1.1*np.max(backward_section), 1.1*np.max(forward_section))
#ax.set_ybound(-1.1*height, .1*height)
ax.set_xbound(-1, 1)
ax.set_ybound(-1, .3)
plt.tight_layout()
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.2)

ray_drawer = drawing.RayDrawer(ax)
segment_drawer = drawing.SegmentDrawer(
    ax,
    color=(0, 1, 1),
    draw_norm_arrows = False,
    norm_arrow_length = .1
)

focal_plane = np.array(
    [
        [
            -max_target_size,
            -height,
            max_target_size,
            -height
        ],
        [
            -2.0*max_target_size,
            -height,
            -max_target_size,
            10.0*max_target_size
        ],
        [
            max_target_size,
            -height,
            2.0*max_target_size,
            10.0*max_target_size
        ]
    ], dtype=np.float64)

# build the training source
angular_distribution = sources.StaticLambertianAngularDistribution(
    -source_angular_cutoff,
    source_angular_cutoff,
    ray_count
)
beam_distribution = sources.StaticUniformBeam(
    -source_aperature_radius,
    source_aperature_radius,
    source_diameter_sample_count
)
training_source = sources.AngularSource(
    (0.0, 0.0),
    central_angle,
    angular_distribution,
    beam_distribution,
    [drawing.YELLOW],
    ray_length=10.0
)

# build the printing source
angular_distribution = sources.StaticLambertianAngularDistribution(
    -source_angular_cutoff,
    source_angular_cutoff,
    5
)
beam_distribution = sources.StaticUniformBeam(
    -source_aperature_radius,
    source_aperature_radius,
    3
)
printing_source = sources.AngularSource(
    (0.0, 0.0),
    central_angle,
    angular_distribution,
    beam_distribution,
    [drawing.YELLOW],
    ray_length=10.0
)

# build the manual source
angular_distribution = sources.StaticLambertianAngularDistribution(
    -source_angular_cutoff,
    source_angular_cutoff,
    10
)
manual_source = sources.PointSource(
    (0.0, 0.0),
    central_angle,
    angular_distribution,
    [drawing.YELLOW],
    ray_length=10.0
)

# package the sources into a single op with tf.case
source_rays = tf.case(
    [
        (
            tf.equal(source_selector, "training"),
            lambda: training_source.rays
        ),
        (
            tf.equal(source_selector, "printing"),
            lambda: printing_source.rays
        ),
        (
            tf.equal(source_selector, "inspection"),
            lambda: manual_source.rays
        )
    ],
    exclusive=True
)

def polynomial(name, max_order, min_order=0, dtype=tf.float64):
    """
        Generates a parametrized polynomial, for use in interpolating stuff.
        Automatically creates a tf variable to store its parametrization.
        
        x : float tensor
            The independant variable used to evaluate the polynomial.
        max_order : int
            The maximum power used in the polynomial.
        min_order : int
            The minumum power used in the polynomial.
            
        returns:
        y : float tensor
            The value of the polynomial, evaluated at x.
        coefficients : tf.Variable
            A handle to the tf.Variable created that stores the parametrization of
            this polynomial.
    """
    with tf.name_scope(name):
        coefficients = tf.get_variable(
            f"{name}_coefficients",
            dtype=dtype,
            shape=(max_order + 1 - min_order),
            initializer=tf.zeros_initializer
        )
        coefficient_stack = tf.unstack(coefficients)
        unused_coefficients = [tf.zeros((), dtype=dtype) for i in range(min_order)]
        
        # tf.math.polyval requires its coefficients be in the reversed order from
        # what I would expect.  After reversing coefficients should be in ascending
        # order.
        all_coefficients = [x for x in 
            reversed(unused_coefficients + coefficient_stack)
        ]
        
        # coefficients are backwards, unfortunately
        y = tf.math.polyval(
            all_coefficients,
            np.linspace(-1, 1, lens_segment_count+1))
        
        return y, coefficients

# build the lens
anchor_angles = np.linspace(
    -lens_angular_cutoff,
    lens_angular_cutoff,
    lens_segment_count+1
)
anchor_angles += central_angle
anchor_x = tf.cos(anchor_angles)
anchor_y = tf.sin(anchor_angles)

first_r, first_parameters = polynomial(
    "first_surface",
    surface_polynomial_order,
    1
)
second_r, second_parameters = polynomial(
    "second_surface",
    surface_polynomial_order,
    1
)

# add a constant term, to keep the center of the lens at a fixed location
first_r += lens_first_fixed_point
second_r += lens_second_fixed_point

# clip r, to prevent the lens from curving around toward zero
first_r = tf.clip_by_value(first_r, first_clip_minus, first_clip_plus)
second_r = tf.clip_by_value(second_r, second_clip_minus, second_clip_plus)
    
first_x = first_r * anchor_x
first_y = first_r * anchor_y
second_x = second_r * anchor_x
second_y = second_r * anchor_y

first_segments = tf.stack([
    first_x[:-1],
    first_y[:-1],
    first_x[1:],
    first_y[1:],
    tf.ones((lens_segment_count,), dtype=tf.float64),
    tf.zeros((lens_segment_count,), dtype=tf.float64)
], axis=1)
second_segments = tf.stack([
    second_x[1:],
    second_y[1:],
    second_x[:-1],
    second_y[:-1],
    tf.ones((lens_segment_count,), dtype=tf.float64),
    tf.zeros((lens_segment_count,), dtype=tf.float64)
], axis=1)
both_segments = tf.concat([first_segments, second_segments], axis=0)

# package the lens surfaces into a selector with tf.case
segments = tf.case(
    [
        (
            tf.equal(segment_selector, "both"),
            lambda: both_segments
        ),
        (
            tf.equal(segment_selector, "first"),
            lambda: first_segments
        )
    ],
    exclusive=True
)

# Set up the ray tracer
reacted_rays, active_rays, finished_rays, dead_rays, _ = ray_tracer.rayTrace(
    source_rays,
    segments,
    None,
    focal_plane,
    None,
    [materials.vacuum, lens_material],
    3
    )
"""
# setup the optimizer
with tf.name_scope("globalStep") as scope:
    global_step = tf.Variable(0.0, trainable=False)
    reset_global_step = tf.assign(global_step, 0.0)
    update_global_step = tf.assign_add(global_step, 1.0)
    
with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.MomentumOptimizer(
        1.0,
        momentum,
        use_nesterov=True
    )
    
def paramatrized_target(finished_ranks, target_definition):
    start, stop, _ = tf.unstack(target_definition)
    return start + finished_ranks * (stop - start)

angle_ranks = training_source.angle_ranks
angle_ranks -= tf.reduce_min(angle_ranks)
angle_ranks /= tf.reduce_max(angle_ranks)

finished_ranks = tf.gather(
    angle_ranks,
    tf.cast(finished_rays[:,5], tf.int64)
)
target_x = paramatrized_target(
    finished_ranks,
    target_definition[target_selector]
)
output_error = tf.squared_difference(target_x, finished_rays[:,2])
output_error_mean = tf.reduce_mean(output_error)

# compute and process the gradients
grads_and_vars = optimizer.compute_gradients(
    output_error,
    [first_parameters, second_parameters]
)
grad_0 = grads_and_vars[0][0]
grad_0 = tf.where(tf.is_finite(grad_0), grad_0, tf.zeros_like(grad_0))
grad_0 = tf.clip_by_value(grad_0, -gradient_clip, gradient_clip)
grad_1 = grads_and_vars[1][0]
grad_1 = tf.where(tf.is_finite(grad_1), grad_1, tf.zeros_like(grad_1))
grad_1 = tf.clip_by_value(grad_1, -gradient_clip, gradient_clip)

# give each surface its own leaning rate
grad_0 = grad_0 * first_LR
grad_1 = grad_1 * second_LR

apply_grads = optimizer.apply_gradients(
    [(grad_0, first_parameters), (grad_1, second_parameters)],
    global_step
)
"""
def update_display(session):
    ray_drawer.rays = np.concatenate(
        session.run(
            [reacted_rays, active_rays, finished_rays, dead_rays],
            context.get_feeds()
        )
    )
    ray_drawer.draw()
    
    segment_drawer.segments = session.run(
        segments,
        context.get_feeds()
    )
    segment_drawer.draw()
    
    drawing.redraw_current_figure()

"""def training_step_first(
    session,
    first_learning_rate,
    momentum,
    central_angle,
    target_definition
):
    error, gs, _ = session.run(
        [first_output_error_mean, global_step, apply_first_grads],
        feed_dict={
            first_LR: first_learning_rate,
            momentum_placeholder: momentum,
            central_angle_placeholder: central_angle,
            target_definition_placeholder: target_definition
        }
    )
    print("Optimization step {0:3d}.  Error: {1:0.5e}".format(int(gs), error))
    
def training_step_second(
    session,
    second_learning_rate,
    momentum,
    central_angle,
    target_definition
):
    error, gs, _ = session.run(
        [second_output_error_mean, global_step, apply_second_grads],
        feed_dict={
            second_LR: second_learning_rate,
            momentum_placeholder: momentum,
            central_angle_placeholder: central_angle,
            target_definition_placeholder: target_definition
        }
    )
    print("Optimization step {0:3d}.  Error: {1:0.5e}".format(int(gs), error))
    
def training_routine(
    session,
    central_angle,
    first_target_definition,
    second_target_definition,
    slice_angle):
    for j in range(250):
        training_step_first(
            session,
            0.001,
            0.9,
            central_angle,
            first_target_definition
        )
        if VISUALIZE and j % VISUALIZE_UPDATE_FREQUENCY == 0:
            update_display_first(session, central_angle, slice_angle)
    for j in range(250):
        training_step_first(
            session,
            0.0001,
            0.9,
            central_angle,
            first_target_definition
        )
        if VISUALIZE and j % VISUALIZE_UPDATE_FREQUENCY == 0:
            update_display_first(session, central_angle, slice_angle)
    session.run(reset_global_step)
    for j in range(250):
        training_step_second(
            session,
            0.001,
            0.9,
            central_angle,
            second_target_definition
        )
        if VISUALIZE and j % VISUALIZE_UPDATE_FREQUENCY == 0:
            update_display_second(session, central_angle, slice_angle)
    for j in range(250):
        training_step_second(
            session,
            0.0001,
            0.9,
            central_angle,
            second_target_definition
        )
        if VISUALIZE and j % VISUALIZE_UPDATE_FREQUENCY == 0:
            update_display_second(session, central_angle, slice_angle)
    session.run(reset_global_step)"""

# ====================================================================================

def print_header():
    # Write the KeyCreator macro file that will generate the set of slices.
    filename = f"TennisCourtSlices.kxl"
    data_out = [
        "//all coordinates are relative to the optic center\r\n",
        f"//height above focal plane: {height}\r\n",
        f"//horizontal offset to beam center: {horizontal_offset}\r\n",
        f"//beam angle: {beam_angles[0]}\r\n",
        f"//beam width +x from center: {beam_width_forward}\r\n",
        f"//beam width -x from center: {beam_width_forward}\r\n",
        f"//beam width +y from center: {beam_width_sideways_forward}\r\n",
        f"//beam width +y from center: {beam_width_sideways_forward}\r\n",
        (f"//source angle cutoff used to design this lens: " + 
        f"{source_angular_cutoff}\r\n"),
        f"//lens angular cutoff: {lens_angular_cutoff}\r\n",
        f"//segments in lens: {lens_segment_count}\r\n",
        (f"//distance from lens center inner surface to optic center: " + 
        f"{lens_first_fixed_point}\r\n"),
        (f"//distance from lens center outer surface to optic center: " + 
        f"{lens_second_fixed_point}\r\n"),
        f"//lines for slice at : {slice_angle} degrees:\r\n",
        f"\r\n"
    ]
    return filename, data_out
    
def print_slice(data_out, slice_angle, kxl_level_counter):
    # extract the solution, and rotate it in 3D to its desired in-application
    # orientation
    solved_segments = session.run(
        segments,
        feed_dict={central_angle: beam_angles[slice_angle]}
    )
    
    solved_start_points = np.pad(
        solved_segments[:,:2],
        ((0,0), (0,1)),
        "constant"
    )
    solved_end_points = np.pad(
        solved_segments[:,2:4],
        ((0,0), (0,1)),
        "constant"
    )
    
    # rotate about Y to form the slices
    rotation = (
        Rotation.from_euler(
            'z',
            90 + np.degrees(beam_angles[0]),
            degrees=True
        ) *
        Rotation.from_euler('Y', slice_angle, degrees=True) *
        Rotation.from_euler(
            'z',
            -90 - np.degrees(beam_angles[slice_angle]),
            degrees=True
        )
    )
    
    solved_start_points = rotation.apply(solved_start_points)
    solved_end_points = rotation.apply(solved_end_points)
        
    rotated_segments = np.concatenate(
        (solved_start_points, solved_end_points),
        axis=1
    )
    
    # write the segments
    data_out.append(f"CLEAR hLevel{kxl_level_counter}\r\n")
    data_out.append(f'LEVEL hLevel{kxl_level_counter}, , ' +  
        f'"slice_{slice_angle}", {kxl_level_counter}\r\n'
    )
    for x1, y1, z1, x2, y2, z2 in rotated_segments:
        # key creater seems to use a different convention for it axes, so
        # I am permuting z and y
        data_out.append(f"LINE {x1}, {z1}, {y1}, {x2}, {z2}, {y2}, , " +
            f"hLevel{kxl_level_counter}, 0\r\n"
        )
    
    # rotate the sparse rays    
    sparse_rays = np.concatenate(
        session.run(
            [sparse_reacted, sparse_active, sparse_finished, sparse_dead],
            feed_dict={central_angle: beam_angles[slice_angle]}
        )
    )
    ray_start_points = np.pad(
        sparse_rays[:,:2],
        ((0,0), (0,1)),
        "constant"
    )
    ray_end_points = np.pad(
        sparse_rays[:,2:4],
        ((0,0), (0,1)),
        "constant"
    )
    ray_start_points = rotation.apply(ray_start_points)
    ray_end_points = rotation.apply(ray_end_points)
        
    rotated_rays = np.concatenate(
        (ray_start_points, ray_end_points),
        axis=1
    )
    
    # write the sparse rays
    for x1, y1, z1, x2, y2, z2 in rotated_rays:
        # key creater seems to use a different convention for it axes, so
        # I am permuting z and y
        data_out.append(f"LINE {x1}, {z1}, {y1}, {x2}, {z2}, {y2}, , " +
            f"hLevel{kxl_level_counter}, 3\r\n"
        )
        
    data_out.append("\r\n")
    kxl_level_counter += 1
    
    return kxl_level_counter

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    # build slice_button
    labels = list(context.get_valid_states_from_context("slice_angle"))
    active = labels.index(context.get_active_state('slice_angle'))    
    slice_button_ax = plt.axes([.05, .25, .2, .7], frameon=False)
    slice_button_ax.text(.1, .95, "slice angles")
    slice_buttons = mpl.widgets.RadioButtons(
        slice_button_ax,
        labels,
        active=active
        )
    def click_slice(label):
        context.set_state("slice_angle", float(label))
        update_display(session)
    slice_buttons.on_clicked(click_slice)
    
    # stuff
    if PRINT:
        filename, data_out = print_header()
        kxl_level_counter = 1
        
    drawing.redraw_current_figure()
    
    """for slice_angle in [0]:
    for slice_angle in slice_angles[::-1]:
        print(f"starting slice at {slice_angle} degrees")
        training_routine(
            session,
            beam_angles[slice_angle],
            first_target_definition_dict[slice_angle],
            second_target_definition_dict[slice_angle],
            slice_angle
        )
        
        if PRINT:
            kxl_level_counter = print_slice(data_out, slice_angle, kxl_level_counter)
            
        print(f"ended slice at {slice_angle} degrees")"""
    if PRINT:
        with open(filename, 'w') as out_file:
            out_file.writelines(data_out)  
    if VISUALIZE:
        update_display(session)
        plt.show()
        
    context.save(context_file_name)
    
    
    
    
    
    
    
        
