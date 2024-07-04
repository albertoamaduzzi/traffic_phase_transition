import pandas as pd
import os
import numpy as np
base_dir = '/home/alberto/LPSim/LivingCity/berkeley_2018/'
dirs = ['boston','los_angeles','rio','lisbon','san_francisco']
work_file = [os.path.join(base_dir,d,'edges.csv') for d in dirs]

for file in work_file:
    df = pd.read_csv(file)
    u = np.unique(df['u'].values)
    utoidx = {u[i]:i for i in range(len(u))}
    df['u'] = df['u'].apply(lambda x:utoidx[x])
    df['v'] =df['v'].apply(lambda x:utoidx[x])
    print(file)
    print(df)
    df.to_csv(file,index = False)