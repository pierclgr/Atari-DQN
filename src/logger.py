from abc import ABC, abstractmethod

import wandb
from src.utils import random_string


class Logger(ABC):
    """
    Abstract class defining a logger
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Method that initializes the logger

        :return: None
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """
        Method closes and finishes the logger

        :return: None
        """
        pass

    @abstractmethod
    def log(self, metric: str, value: float, step: int) -> None:
        """
        Method that logs the metrics given in input

        :param metric: the metric we're logging (str)
        :param value: the value of the metric we are logging at this time (float)
        :param step: the current time at which we're logging the metric value (int)

        :return: None
        """
        pass


class WandbLogger(Logger):
    """
    Class that describes a Wandb logger, subclass of Logger
    """

    def __init__(self,
                 name: str,
                 config: dict,
                 project: str = "AAS_project",
                 entity: str = "pierclgr",
                 monitor_gym: bool = True) -> None:
        """
        Constructor method of a Wandb logger

        :param name: the name of the Wandb run in which we're logging (str)
        :param config: the configuration of the current run to be logged (dict)
        :param project: the name of the Wandb project on which to log, default value is "AAS_project" (str)
        :param entity: the name of the entity on Wandb on which to log, default value is "pierclgr" (str)
        :param monitor_gym: a boolean defining if we're monitoring a gym environment to also log videos outputted by
            gym environment, default value is True (bool)

        :return: None
        """

        self.name = name + "_" + random_string()
        self.project = project
        self.entity = entity
        self.config = config
        self.monitor_gym = monitor_gym

        self.logger = wandb.init(project=self.project, name=self.name, entity=self.entity, config=self.config,
                                 monitor_gym=self.monitor_gym)

    def finish(self) -> None:
        """
        Method closes and finishes the Wandb logger

        :return: None
        """
        self.logger.finish()

    def log(self, metric: str, value: float, step: int) -> None:
        """
        Method that logs the metrics to the define run

        :param metric: the metric we're logging (str)
        :param value: the value of the metric we are logging at this time (float)
        :param step: the current time at which we're logging the metric value (int)

        :return: None
        """

        # log the input metric to the Wandb run
        self.logger.log({metric: value}, step=step)
