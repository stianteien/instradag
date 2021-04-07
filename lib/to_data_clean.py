# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:17:58 2021

@author: Stian
"""

import os
import pandas as pd
import numpy as np
from make_ready import make_ready

filer = os.listdir('../data')

for fil in filer:
    df = make_ready().use_stockstats(['../data/'+fil])[0]
    df.to_excel('../data_clean/'+fil, index=False)
    print(f"Gjort om to data_clean: {fil}")
