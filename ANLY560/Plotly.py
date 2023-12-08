#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:44:07 2022

@author: jieyisun
"""
import chart_studio
import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objects as go #has more control, customizable
import plotly.io as pio #produce an html file
import plotly.express as px #fast, low effort
from datetime import datetime


sp="Covid-19SG.csv"
SP=pd.read_csv(sp)

print(SP.head())
print(SP.dtypes)
SP['TrueDate']=(pd.to_datetime(SP['Date'].astype(str), format='%Y/%m/%d'))
print(SP.head())
print(SP.dtypes)

SPfig1 = go.Figure()
SPfig1.add_trace(go.Scatter(x=SP["TrueDate"], y=SP["Daily Confirmed"],
                            mode='lines',
                            name='Daily Confirmed'))
SPfig1.update_layout(
    title='Daily Confirmed',
    yaxis_title='Daily Confirmed',
    xaxis_title='Time')

pio.write_image(SPfig1, 'test.png')