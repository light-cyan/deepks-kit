import pytest
from pyscf import lib


@pytest.fixture(scope="session", autouse=True)
def use_single_pyscf_thread():
    original_thread_count = lib.num_threads()
    lib.num_threads(1)
    try:
        yield
    finally:
        lib.num_threads(original_thread_count)
