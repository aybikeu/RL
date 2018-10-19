import pandas as pd



df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/pr_for_INS3.csv')
epsilon = 0.1
convergence_check=False

#n_states is 862 for Instance 2 - 1053 for instance 1
n_states = int(df['s_prime'].sort_values(ascending=False).values[0] + 1)

n_actions= 17
n_sas = df.shape[0] #number of all (s, a, s_prime)

Q_T = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)
Q_Tnext = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)

while convergence_check==False:
#for _ in range(20):
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

    if Q_diff.sum().sum() <= 0.5:
        convergence_check = True

Q_Tnext.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS3.csv',sep=',')



