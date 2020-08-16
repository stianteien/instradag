'''
16.08.2020
Stian

This file is used for cleaning up excel fil
and make data from the last day registered
'''

import pandas as pd


class rens:
    def __init__(self):
        pass

    def clean_data(self, data):
        self.data = data
        self.data = self.data.rename(columns={self.data.columns[0]: "tid"})
        self.data = self.data.drop(columns=['Kjøper', 'Selger', 'Type'])

        count=0
        this_day = self.data.tid[0].day
        for i in self.data.tid:
            if i.day == this_day:
                count += 1
                
        self.data = self.data[:count]
        self.data = self.data[5:]

        # Flipper opp ned
        self.data = self.data.iloc[::-1]
    
        # Legger til for stockstats
        self.data['open'] = data.Pris
        self.data['close'] = data.Pris
        self.data['high'] = data.Pris
        self.data['low'] = data.Pris
        self.data['volume'] = data.Volum
        self.data['amount'] = data.Volum
        
        self.data = self.data.drop(columns=['Pris', 'Volum'])
        
        return self.data
        
