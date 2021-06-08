import pandas as pd
import sys
import matplotlib.pyplot as plt

df = pd.read_csv(sys.stdin,names=['t','x',r'\dot{x}'],index_col='t')
ax = df.plot()
plt.show()