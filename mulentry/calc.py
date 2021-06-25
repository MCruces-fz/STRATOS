import numpy as np

from mulentry.chef import Chef


class Calc:
    _substracted = None
    _mean_x = None
    _mean_y = None
    _std_x = None
    _std_y = None

    def __init__(self, reader: Chef):
        self.reader = reader

    @property
    def substracted(self):
        if self._substracted is None:
            self._substracted = self.decrease_entries(self.reader.total_entries)
        return self._substracted

    @property
    def mean_x(self):
        if self._mean_x is None:
            self._mean_x = np.mean(self._substracted, axis=0)
        return self._mean_x

    @property
    def mean_y(self):
        if self._mean_y is None:
            self._mean_y = np.mean(self._substracted, axis=1)
        return self._mean_y

    @property
    def std_x(self):
        if self._std_x is None:
            self._std_x = np.std(self._substracted, axis=0)
        return self._std_x

    @property
    def std_y(self):
        if self._std_y is None:
            self._std_y = np.std(self._substracted, axis=1)
        return self._std_y

    @staticmethod
    def decrease_entries(entries: np.array, preserve_size: bool = True) -> np.array:
        """
        Substracts minimum value of entries from inner region (avoiding
            external crown)
    
        :param entries: 2D array to be decreased.
        :param preserve_size: If True (default) returns array with same
            size. If False, returns array with inner reguin (substracting
            external crown)
        :return: 2D array with decreased entries.
        """
    
        if len(entries.shape) != 2:
            raise Exception("Input array must be 2D.")
    
        if preserve_size:
            substracted = entries - np.min(entries[1:-1, 1:-1])
        else:
            substracted = entries[1:-1, 1:-1] - np.min(entries[1:-1, 1:-1])
        return substracted.clip(min=0)

