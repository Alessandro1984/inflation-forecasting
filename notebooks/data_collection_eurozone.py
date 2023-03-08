#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import os.path
import json
import requests
import pandas as pd
from pprint import pprint
from functools import reduce
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


# In[4]:


base_uri = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"

endpoints = [
        'PRC_HICP_MANR/M.RCH_A.CP00.',
        'PRC_HICP_MANR/M.RCH_A.TOT_X_NRG_FOOD.',
        'STS_INPR_M/M.PROD.C.SCA.PCH_PRE.',
        'UNE_RT_M/M.SA.TOTAL.PC_ACT.T.',
        'IRT_ST_M/M.IRT_M1.EA'
    ]

def obtain_data(country):
    
    suffix_country = f'{country}?format=JSON&lang=en'
    full_endpoints = [base_uri + endpoints[i] + suffix_country for i in range(len(endpoints)-1)]
    xtra_endpoint = base_uri + endpoints[-1] + '?format=JSON&lang=en'
    full_endpoints.append(xtra_endpoint)
    
    dfs = {}
    
    for idx, url in enumerate(full_endpoints):
        
        response = requests.get(url)
        
        assert response.status_code == 200, "Bad request"
        
        years = response.json()['dimension']['time']['category']['label']
        value = response.json()['value']
        df1 = pd.DataFrame.from_dict(years, orient='index', columns=['Year']).reset_index()[['Year']]
        df1 = df1.rename_axis('Id').reset_index()
        df2 = pd.DataFrame.from_dict(value, orient='index').rename(columns={0: "Value"})
        df2 = df2.rename_axis('Id').reset_index().copy()
        df2 = df2.astype({'Id': 'int64'}).copy()
        df_merged = pd.merge(df1, df2, on = "Id")
        df = df_merged[['Year', 'Value']]

        dfs[idx] = df
        
    df_cpi = dfs[0]
    df_cpi = df_cpi.rename(columns={"Value": "cpi"})
    df_ccpi = dfs[1]
    df_ccpi = df_ccpi.rename(columns={"Value": "ccpi"})
    df_indprod = dfs[2]
    df_indprod = df_indprod.rename(columns={"Value": "indprod"})
    df_unemp = dfs[3]
    df_unemp = df_unemp.rename(columns={"Value": "unemp"})
    df_intrate = dfs[4]
    df_intrate = df_intrate.rename(columns={"Value": "intrate"})
    
    df_additional = pd.read_csv("raw_data/data_additional.csv", sep = ";")

    df_merged = df_cpi.merge(df_ccpi, how='left').merge(df_indprod, how='left').merge(df_unemp, how='left').merge(df_intrate, how='left').merge(df_additional, how='left')
    
    country_id  = []

    for x in range(df_merged.shape[0]):
        country_id.append(country)

    df_country_id = pd.DataFrame(country_id, columns=['country_id'])

    df_final = df_merged.join(pd.DataFrame(df_country_id))
    
    return df_final


# In[5]:


countries = ['IT', 'DE', 'NL', 'ES', 'FR']
df_countries = [obtain_data(country) for country in countries]
df_all_new = pd.concat(df_countries, ignore_index=True)


# In[18]:


df_all_new


# In[ ]:


#sns.heatmap(df_all.drop(['Year', 'country_id'], axis=1).corr(), annot=True);


# In[19]:


df_all_new.to_csv("raw_data/data_euro.csv", index = False)

