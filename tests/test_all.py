# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:57:00 2020

@author: Stian

This is a test file. Run this file to test new editfukcs
"""

import pandas as pd
import numpy as np
import os
import pytest
import sys

#from .lib import make_ready
#from ..lib.rens import rens

#data = pd.read_excel('../data/Aker 02.04.2020.xlsx')
filnavn = '../data/Aker 02.04.2020.xlsx'

# Rense
@pytest.fixture
def test_rens():    
    
    #data = rens().clean_data(filnavn)
    assert 3==3


test_rens()
## Test make_ready
#stocks = make_ready().use_stockstats(data, )

