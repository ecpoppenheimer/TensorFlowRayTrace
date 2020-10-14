import tensorflow as tf

import tfrt.optimizer as op

parameters = [
    tf.Variable((1, 2, 3, 4)),
    tf.Variable((10, 20, 30, 40))
]
optimizer = op.SGD_Optimizer(None, parameters, None, None)
routine = [
    {"steps": 5},
    {"steps": 10},
    {"steps": 5}
]
optimizer.training_routine(routine, report_frequency=1)
