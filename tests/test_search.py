import importlib


def test_search_importable():
    assert importlib.import_module("task_a_pathfinding.search") is not None
