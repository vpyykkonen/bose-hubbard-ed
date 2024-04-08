import os
import numpy as np

n_ts = 10
ts = np.linspace(0.1,10,n_ts)

for t in ts:
    os.system('python3 bose_hubbard_ed.py diamond edge_site_left 2 1.0 2 3.14159265 '+ str(t) +' 0 3000 3001')
