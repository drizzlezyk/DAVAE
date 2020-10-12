import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

base_path = '/Users/zhongyuanke/data/'

df = pd.DataFrame({
'RAW': [0,0,0,0],
'DAVAE': [0.27,0.38,0.28,0.17],
'Scanorama': [0.01, 0.01, 0.03, 0],
"DESC": [0, 0, 0, 0],
"Seurat 3.0": [0, 0, 0, 0]
})

df.boxplot(grid=False,  fontsize=10, )
plt.title('kBET')
plt.ylabel('acceptance rate')
plt.savefig(base_path+'dann_vae/k_bet/293t_kbet.png')


