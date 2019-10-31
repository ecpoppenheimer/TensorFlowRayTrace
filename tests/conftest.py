import pytest

import matplotlib.pyplot as plt
import tensorflow as tf


@pytest.fixture(scope="function")
def session():
    tf.reset_default_graph()
    with tf.Session() as session:
        yield session
