from abc import ABC, abstractmethod

import numpy as np


class RlModel(ABC):
    @abstractmethod
    def __init__(self, state_shape: tuple[int], action_shape: tuple[int], **kwargs):
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def add_to_buffer(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: np.ndarray):
        pass

    # clientのモデルのうち、global共有したい部分だけを抜き出す
    # @abstractmethod
    # def get_checkpoint(self) -> RlModel:
    #     pass
