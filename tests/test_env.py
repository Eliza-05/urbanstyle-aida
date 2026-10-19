import importlib


def test_env_importable():
    assert importlib.import_module("task_c_pricing.env") is not None
