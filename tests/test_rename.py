# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:06:29 2020

@author: Stian

test rename and make new file
"""

import pytest
import numpy as np
import pandas as pd
import os

from lib.rename_data import rename_data
from lib.rename_data_euronext import rename_data_euronext

# =============================================================================
# Make dummy data
# =============================================================================

def test_make_dummy(): #vil ikke lagre ssseeeg!
    dummy_data = np.array([['2020-12-14 16:04',"517.5","299"],
                        ['2020-12-14 16:06',"518.5","226"],
                        ['2020-12-14 16:07',"518","356"],
                        ['2020-12-14 16:08',"518","255"],
                        ['2020-12-14 16:09',"518.5","82"],
                        ['2020-12-14 16:13',"519","301"],
                        ['2020-12-14 16:16',"520","8"],
                        ['2020-12-14 16:17',"520","30"],
                        ['2020-12-14 16:18',"520.5","340"],
                        ['2020-12-14 16:19',"519","1105"],
                        ['2020-12-14 16:25',"521","19143"]])
    
    data = pd.DataFrame(dummy_data, columns=['time',"dummy","dummy( volume )"])
    data.to_csv('../tests/dummy_quote_data.csv', index=False)
    #assert 2==3


# =============================================================================
# Testing
# =============================================================================

# New file renameee
def test_rename_euronext():
    re = rename_data_euronext()
    re.path = '../tests/'
    re.files = ['dummy_data.csv']
    
def test_rename_euronext_rename():
    re = rename_data_euronext()
    re.path = '../tests/'
    re.files = ['dummy_quote_data.csv']
    re.rename()



# =============================================================================
# Delete dummy data
# =============================================================================

def test_remove_file():
    os.remove('../tests/dummy 14.12.2020.xlsx')
