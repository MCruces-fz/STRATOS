import numpy as np
import warnings

from mulentry.chef import Chef
from utils.footilities import decrease_entries, raw_moment, intertial_axis, main_direction


class Calc:
    _substracted = None
    _mean_x = None
    _mean_y = None
    _std_x = None
    _std_y = None

    xbar, ybar, cov = None, None, None
    eigvals, eigvecs = None, None
    main_dir = None
    lenght_main = None
    angle = None
    angle_45 = None


    def __init__(self, reader: Chef):
        self.update(reader)

    @property
    def substracted(self):
        if self._substracted is None:
            self._substracted = self.decrease_entries(self.reader.total_entries)
        return self._substracted

    @substracted.setter
    def substracted(self, subs):
        if subs is None:
            warnings.warn("Assigning None to substracted by user.", stacklevel=2)
        self._substracted = subs

    @property
    def mean_x(self):
        if self._mean_x is None:
            self._mean_x = np.mean(self._substracted, axis=0)
        return self._mean_x

    @mean_x.setter
    def mean_x(self, mx):
        if mx is None:
            warnings.warn("Assigning None to mean_x by user.", stacklevel=2)
        self._mean_x = mx

    @property
    def mean_y(self):
        if self._mean_y is None:
            self._mean_y = np.mean(self._substracted, axis=1)
        return self._mean_y

    @mean_y.setter
    def mean_y(self, my):
        if my is None:
            warnings.warn("Assigning None to mean_y by user.", stacklevel=2)
        self._mean_ya = my

    @property
    def std_x(self):
        if self._std_x is None:
            self._std_x = np.std(self._substracted, axis=0)
        return self._std_x

    @std_x.setter
    def std_x(self, sx):
        if sx is None:
            warnings.warn("Assigning None to std_x by user.", stacklevel=2)
        self._std_x = sx

    @property
    def std_y(self):
        if self._std_y is None:
            self._std_y = np.std(self._substracted, axis=1)
        return self._std_y

    @std_y.setter
    def std_y(self, sy):
        if sy is None:
            warnings.warn("Assigning None to std_y by user.", stacklevel=2)
        self._std_y = sy

    def update(self, reader: Chef = None):
        """
        Update method to refresh all values.

        :param reader: (Optional) New input data.
        """

        if reader is not None:
            self.reader = reader

        self.substracted = decrease_entries(self.reader.total_entries)

        self.mean_x = np.mean(self.substracted, axis=0)
        self.mean_y = np.mean(self.substracted, axis=1)
        self.std_x = np.std(self.substracted, axis=0)
        self.std_y = np.std(self.substracted, axis=1)

        self.xbar, self.ybar, self.cov = intertial_axis(reader.total_entries)
        self.eigvals, self.eigvecs = np.linalg.eigh(self.cov)
        self.main_dir = main_direction(self.eigvals, self.eigvecs)
        self.lenght_main = np.sqrt(self.main_dir[0]**2 + self.main_dir[1]**2)
        self.angle = np.arctan(self.main_dir[1] / self.main_dir[0])
        self.angle_45 = self.angle - np.pi/4


