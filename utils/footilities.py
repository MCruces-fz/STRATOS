"""
             A P A C H E   L I C E N S E
                    ------------ 
              Version 2.0, January 2004

       Copyright 2021 Miguel Cruces Fernández

  Licensed under the Apache License, Version 2.0 (the 
"License"); you may not use this file except in compliance 
with the License. You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied. See the License for the specific 
language governing permissions and limitations under the 
License.

           miguel.cruces.fernandez@usc.es
               mcsquared.fz@gmail.com
"""

import os
import datetime
import numpy as np


def basename(file_path: str, extension: bool = False):
    """
    Get the basename of the full path to the file.

    :param file_path: Full path to the file.
    :param extension: If returns filename with extension or not.
    """
    file_name = os.path.basename(file_path)
    if not extension:
        file_name, *_ = file_name.split(".")
    return file_name


def date_from_filename(filename: str) -> datetime.datetime:
    """
    Return datetime.datetime instance from any filename as
    tryydoyhhmmss.hld.root.root or styydoyhhmmss.hld.root.root

    :param filename: Filename like tryydoyhhmmss.hld.root.root or
        styydoyhhmmss.hld.root.root.
    :return: datetime.datetime obkect.
    """

    if not filename.startswith(("st", "tr")) or not filename.endswith(".hld.root.root"):
        raise Exception("Filename must be like tryydoyhhmmss.hld.root.root "
                        "or styydoyhhmmss.hld.root.root")

    yy = int(f"20{filename[2:4]}")
    doy = int(filename[4:7])
    hh = int(filename[7:9])
    mm = int(filename[9:11])
    ss = int(filename[11:13])

    return datetime.datetime.combine(
        datetime.date(yy, 1, 1) + datetime.timedelta(doy + 1),
        datetime.time(hour=hh, minute=mm, second=ss)
    )


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


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()


def intertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov


def main_direction(eigvals,eigvecs):
    main_dir = eigvals[0]*eigvecs[:,0] + eigvals[1]*eigvecs[:,1]
    if main_dir[0] / main_dir[1] < 0:
        main_dir = eigvals[0]*eigvecs[:,0] - eigvals[1]*eigvecs[:,1]
    return main_dir
