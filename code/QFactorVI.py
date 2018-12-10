import pandas as pd
import csv
from collections import defaultdict



df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/pr_for_INS2_V3.csv')
#df = pd.read_csv('/home/ulusan.a/RL/data_files/pr_for_INS2_V3.csv')


reader = csv.reader(open('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/pr_for_INS2_V3.csv', 'r'))
#reader = csv.reader(open('/home/ulusan.a/RL/data_files/pr_for_INS2_V3.csv', 'r'))
next(reader, None)
prob = {}
rew = {}
for row in reader:
    s,a,p,r,s_prime = row
    s = int(float(s))
    a = int(float(a))
    s_prime = int(float(s_prime))

    try:
        prob[(s,a)][s_prime] = p
    except:
        prob[(s, a)] = {}
        prob[(s, a)][s_prime] = p
    try:
        rew[(s,a)][s_prime] = r
    except:
        rew[(s, a)] = {}
        rew[(s, a)][s_prime] = r


epsilon = 0.1
convergence_check=False

#n_states is 862 for Instance 2 - 1053 for instance 1
n_states = int(df['s_prime'].sort_values(ascending=False).values[0] + 1)

n_actions = int(df['a'].sort_values(ascending=False).values[0] + 1)
#n_sas = df.shape[0] #number of all (s, a, s_prime)

Q_T = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)
Q_Tnext = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)
iter = 0
while convergence_check==False:
#for _ in range(20):
    iter += 1
    iter2 = 0

    # for s in range(n_states):
    #     for a in range(n_actions):
    for s,a in prob.keys():
            #df_slice = df.query('s== {} & a=={}'.format(s,a))
        added_value = 0
        #iter2 += 1
        for s_p in prob[(s,a)].keys():
            reward = float(rew[(s,a)][s_p])
            pr = float(prob[(s,a)][s_p])
       # for _, r in df_slice.iterrows():
       #         s_p = int(r['s_prime'])
       #         reward = r['reward']
       #         pr = r['probability']
            added_value = added_value + (pr * (reward + Q_T.iloc[s_p].max()))

        Q_Tnext.iloc[s][a] = added_value
        #print('Inner iter:', iter2)

    Q_diff = Q_Tnext - Q_T
    Q_T = Q_Tnext.copy()
    print('Iteration no:', iter, 'Epsilon:', Q_diff.sum().sum())
    if Q_diff.sum().sum() <= 2:
        convergence_check = True

#Q_Tnext.to_csv('/home/ulusan.a/RL/data_files/Q_optimalVI_INS2_V3_2.csv',sep=',')
Q_Tnext.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS2_V3_2.csv',sep=',')


