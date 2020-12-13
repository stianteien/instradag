'''
16.08.2020
Stian

This file is used for cleaning up excel fil by:
removing dates
just have 1 price per minute
making ready for stockstats
'''

import pandas as pd
#fil = pd.read_excel('../data/test/Equinor 12.11.2020.xlsx')
#fil = pd.read_excel('../data/Aker 02.04.2020.xlsx')

class rens:
    def __init__(self):
        pass

    def clean_data(self, data):
        self.data = data
        self.data = self.data.rename(columns={self.data.columns[0]: "tid"})
        self.data = self.data.drop(columns=['Kjøper', 'Selger', 'Type'])
        if type(self.data.tid[0]) == str:
            self.data['tid'] = pd.to_datetime(self.data['tid'], format='%d.%m.%Y %H:%M:%S')

        # Finner kun fra dagen i dag
        count=0
        this_day = self.data.tid[0].day
        for i in self.data.tid:
            if i.day == this_day:
                count += 1
        
        
        # Henter det jeg skal   
        self.data = self.data[:count]
        
        # Kun ett og ett minutt
        every_min = []
        volume_per_min = {}
        for i,tiden in enumerate(self.data.tid):
            every_min.append(str(tiden.hour) +':'+ str(tiden.minute))
            if every_min[-1] not in volume_per_min:
                volume_per_min[every_min[-1]] = self.data.iloc[i]['Volum']
            else:
                volume_per_min[every_min[-1]] += self.data.iloc[i]['Volum']
                
            
        self.data.tid = every_min
        self.data = self.data.drop_duplicates(subset='tid')
        
        # Legger til volum fra hvert minutt
        self.data.Volum = list(volume_per_min.values())

        # Flipper opp ned
        self.data = self.data.iloc[::-1]
        
        # Resetter indexen
        self.data = self.data.reset_index(drop=True)
    
        # Legger til for stockstats
        self.data['open'] = self.data.Pris
        self.data['close'] = self.data.Pris
        self.data['high'] = self.data.Pris
        self.data['low'] = self.data.Pris
        self.data['volume'] = self.data.Volum
        self.data['amount'] = self.data.Volum
        
        self.data = self.data.drop(columns=['Pris', 'Volum'])
        
        return self.data
    
        
#my = rens()
#a = (my.clean_data(fil))
#print(a.head(10))
#a.to_excel('../data/test.xlsx')