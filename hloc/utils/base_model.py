import sys
import inspect
import logging
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Any

import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super(BaseModel, self).__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError


def dynamic_load(root, model: str) -> BaseModel:
    module_path = f'{root.__name__}.{model}'

    logging.debug(f'Loading module "{module_path}"')
    module = __import__(module_path, fromlist=[''])
    classes: list[tuple[str, Any]] = inspect.getmembers(module, inspect.isclass)

    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]

    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseModel)]

    assert len(classes) == 1, 'There should be exactly one class inherited from BaseModel in the module.'
    return classes[0][1]
