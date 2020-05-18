#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:14:14 2020

@author: mattjaffe
"""

from sloppy.optic import *
from sloppy.raytracing import *
from sloppy.tools import *
from sloppy.abcd import *
from sloppy.utils import *
import sys
sys.path.append('../')
from cavities import *
waists_vs_param(Cav1L0Q, 'lens_dist', 1e-2)