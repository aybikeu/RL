import numpy as np
import pandas as pd

state_info = pd.read_csv('C:\Users\ulusan.a\Desktop\RL_rep\RL\data_files\state_info_100kVI_INS2.csv', header=None)
state_info.columns = ['state feature', 'value']
state_info['state'], state_info['feature'] = state_info['state feature'].str.split(',',1).str
state_info['state']=state_info['state'].str[1:].astype('int64')
state_info['feature']=state_info['feature'].str[1:-1]

state_mapper = {}

max_state = state_info['state'].max()
for state_index in range(max_state+1):
    df_s = state_info.query('state =={}'.format(state_index))
    #first change the str data type to float - make an float array to have accurate hashing
    debris_hash = hash(tuple(np.fromstring(df_s.iloc[0]['value'][1:-1],dtype=float,sep=',')))
    demand_hash = hash(tuple(np.fromstring(df_s.iloc[1]['value'][1:-1],dtype=float,sep=',')))
    supply_hash =  hash(tuple(np.fromstring(df_s.iloc[4]['value'][1:-1],dtype=float,sep=',')))

    state_hash = hash((debris_hash,supply_hash ,demand_hash ))
    state_mapper[state_hash] = state_index

df_map = pd.DataFrame.from_dict(state_mapper, orient='index')
df_map.to_csv('C:\Users\ulusan.a\Desktop\RL_rep\RL\data_files\state_mapper_INS2.csv', header=None)