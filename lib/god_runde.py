'''
17.08.2020
Stian

Hente ut dager der det har gått bra
God runde er en lang kontinuerlig oppgang.

Hvis siste 10 minuttene er bedre enn siste 20 er det en oppgang
'''

import pandas as pd
import math
import matplotlib.pyplot as plt
fil = pd.read_excel('../data/test.xlsx')

#siste_10min = dataframe['Pris'].iloc[i-5:i+1]
#siste_20min = dataframe['Pris'].iloc[i-20:i+1]

class positiv_utvikling:
    def __init__(self):
        pass
    
    def calculate(self, data):
        self.data = data
        self.df_utvikling = pd.DataFrame(columns= ['index', 'endex', 'close', 'score'])
        utvikling = []
        
        for ix, pris in self.data.close.iteritems(): 
            if(ix >= 20):
                siste_10min = self.data.close[ix-10:ix]
                siste_20min = self.data.close[ix-20:ix]
                utvikling.append(sum(siste_10min)/len(siste_10min) - sum(siste_20min)/len(siste_20min))
                
        pos_utvikling = [0]
        for i in utvikling:
            if i > 0:
                pos_utvikling.append(pos_utvikling[-1] + self.score_algo(i))
                
        plt.plot(pos_utvikling)
        #print((utvikling))
        
        return self.data
    
    
    def score_algo(self, score):
        return math.sqrt(score)
    
    
myp = positiv_utvikling()
myp.calculate(fil)