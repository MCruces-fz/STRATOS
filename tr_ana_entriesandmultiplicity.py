#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:17:54 2020

Author: *Martina Feijoo Fontán*
Original script `tr_ana_entriesandmultiplicity.py`
in Martina's fork: 
    https://github.com/fei-martina/STRATOS

EDIT: MCruces-fz
*Miguel Cruces Fernández*
- miguel.cruces.fernandez@usc.es
- mcsquared.fz@gmail.com


--------------------------------------------------
Analysis of TRAGALDABAS entries excluding borders.
--------------------------------------------------
"""

from datetime import datetime
import numpy as np

from utils.dirs import ASCII_DATA_DIR
from utils.footilities import decrease_entries, intertial_axis, main_direction
from mulentry.chef import Chef
from mulentry.calc import Calc


reader = Chef(data_dir=ASCII_DATA_DIR)

reader.update(
        from_date=datetime(
            year=2021,
            month=1,
            day=18,
            # hour=12,  # TODO: Allow time selection
            # minute=15,
            # second=16
        ),
        to_date=datetime(
            year=2021,
            month=1,
            day=18,
            # hour=16,
            # minute=30,
            # second=56
        ),
        plane_name="T1"
        )

# print("multiplic: ", reader.total_mult)
# print("total: ", reader.total_entries)



substracted = decrease_entries(reader.total_entries)
# print("substracted: ", substracted)

mean_x = np.mean(substracted, axis=0)
mean_y = np.mean(substracted, axis=1)
std_x = np.std(substracted, axis=0)
std_y = np.std(substracted, axis=1)

# covariance, eigvals, main direction, etc
xbar, ybar, cov = intertial_axis(reader.total_entries)
eigvals, eigvecs = np.linalg.eigh(cov)
main_dir = main_direction(eigvals, eigvecs)
lenght_main = np.sqrt(main_dir[0]**2 + main_dir[1]**2)
angle = np.arctan(main_dir[1] / main_dir[0])
angle_45 = angle - np.pi/4



