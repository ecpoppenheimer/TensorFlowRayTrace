
import pytest

import matplotlib.pyplot as plt


@pytest.fixture(scope="function")
def ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    yield ax
    plt.close(fig)
