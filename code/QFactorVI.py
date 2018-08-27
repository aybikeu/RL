import pandas as pd



df = pd.read_csv('pr_forVIR2.csv')
epsilon = 0.1
convergence_check=False

n_states = 1053
n_actions=7
n_sas = df.shape[0] #number of all (s, a, s_prime)

Q_T = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)
Q_Tnext = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)

#while convergence_check==False:
for _ in range(5):
    for s in range(n_states):
        for a in range(n_actions):
            df_slice = df.query('s== {} & a=={}'.format(s,a))
            added_value = 0
            for _, r in df_slice.iterrows():
                    s_p = int(r['s_prime'])
                    reward = r['reward']
                    pr = r['probability']
                    added_value = added_value + (pr * (reward + Q_T.iloc[s_p].max()))

            Q_Tnext.iloc[s][a] = added_value

    Q_diff = Q_Tnext - Q_T
    Q_T = Q_Tnext.copy()

if Q_diff.sum().sum() <= epsilon * 3000:
    convergence_check = True

Q_Tnext.to_csv('Q_optimalVIR2.csv',sep=',')



