import os
import gzip
import pickle
import abc


class LoadSave(abc.ABC):
    """
    Inherit from this class and a basic implementation of saving an object using pickling is added to the class
    """

    def save(self, filename: str):
        with gzip.open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath):
        with gzip.open(filepath, "rb") as file:
            return pickle.load(file)
