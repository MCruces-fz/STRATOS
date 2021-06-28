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
from mulentry.represent import Represent


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

calc = Calc(reader)

represent = Represent(calc)

