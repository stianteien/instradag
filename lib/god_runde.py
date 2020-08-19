'''
17.08.2020
Stian

Hente ut dager der det har gått bra
God runde er en lang kontinuerlig oppgang.

Hvis siste 10 minuttene er bedre enn siste 20 er det en oppgang
'''

import pandas as pd
fil = pd.read_excel('../data/test.xlsx')

#siste_10min = dataframe['Pris'].iloc[i-5:i+1]
#siste_20min = dataframe['Pris'].iloc[i-20:i+1]

class positiv_utvikling:
    def __init__(self):
        pass
    
    def calculate(self, data):
        self.data = data
        
        for ix, pris in self.data.close.iteritems(): #.close
            if(ix >= 20):
                siste_10min = self.data.close[ix-10:ix+1]
                siste_20min = self.data.close[ix-20:ix+1]
                
        print(siste_10min)
        
        return self.data
    
    
myp = positiv_utvikling()
myp.calculate(fil)