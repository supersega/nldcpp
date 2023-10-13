#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:33:33 2020

@author: supersega
"""

import matplotlib.pyplot as plt
import numpy as np

x, y = np.loadtxt('curve.txt', unpack = True, delimiter = ' ')

plt.plot(x, y, color = 'coral')
plt.savefig('curve.pdf')

