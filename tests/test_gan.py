import importlib


def test_gan_importable():
    assert importlib.import_module("task_d_gan.models") is not None
