import importlib


def test_data_importable():
    assert importlib.import_module("common.data") is not None
