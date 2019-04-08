import pytest

import tensorflow as tf


@pytest.fixture(scope="function")
def session():
    session = tf.Session()
    yield session
    session.close()
