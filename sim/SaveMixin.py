import os
import pickle
import abc


class SaveMixin(abc.ABC):
    """
    Inherit from this class and a basic implementation of saving an object using pickling is added to the class
    """

    @abc.abstractmethod
    def get_filename(self):
        pass

    def save(self, directory: str, suffix=""):
        # If there is no world_cache directory, create it
        with open(f"{directory}.pickle", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
