import os
import pickle
import abc


class LoadSave(abc.ABC):
    """
    Inherit from this class and a basic implementation of saving an object using pickling is added to the class
    """

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
