import pandas as pd
import sys
import matplotlib.pyplot as plt

df = pd.read_csv(sys.stdin,names=['t','x',r'\dot{x}',r"dx_dx0", r"d\dot{x}_dx0", r"dx_d\dot{x}0", r"d\dot{x}_d\dot{x}0", r"dx_dk", r"d\dot{x}_dk"],index_col='t')
ax = df.plot(subplots=True)
plt.suptitle("\dotdot{x} = -k*x")
plt.show()