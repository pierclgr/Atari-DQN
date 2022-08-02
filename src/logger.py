from abc import ABC, abstractmethod

import wandb
from src.utils import random_string


class Logger(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    @abstractmethod
    def log(self, metric: str, value: float, step: int):
        pass


class WandbLogger(Logger):
    def __init__(self,
                 name: str,
                 config: dict,
                 cache_size: int = 100,
                 project: str = "AAS_project",
                 entity: str = "pierclgr"):
        self.name = name + "_" + random_string()
        self.project = project
        self.entity = entity
        self.config = config
        self.cache_size = cache_size

        self.logger = wandb.init(project=self.project, name=self.name, entity=self.entity, config=self.config)

        self.metrics = {}

    def finish(self):
        self.logger.finish()

    def log(self, metric: str, value: float, step: int):
        self.logger.log({metric: value}, step=step)
