# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:42:41 2022

@author: steffejb
"""

import pandas as pd

df = pd.read_csv ('output.csv',sep=';',names=['run',	'Instance',	'Analyses',	'trips',
                                      'starvations','congestions','starvation_std'	,'congestion_std'])

df['violations'] = df['starvations'] + df['congestions'] 

df = df.sort_values(by=['violations'],ascending=True)
