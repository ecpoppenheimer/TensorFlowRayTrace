import pytest

import tensorflow as tf


@pytest.fixture(scope="module")
def session():
    with tf.Session() as session:
        yield session
