
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

repopath = Path(__file__).parent
fname = r'1632344375 - two towers and noise.asc'
file = os.path.join(repopath,fname)
output = os.path.join(repopath,'beta_angles_' + fname)

#create pandas dataframe from asc file
df = pd.read_csv(file,header=None,delimiter = ' ')

#rename the columns based on output from cloudcompare 
# df.columns =['x','y','z', 'dist','ang','time','r','g','b']
df.columns = ['x', 'y', 'z', 'dist', 'r', 'g', 'b','ang', 'time']


#prelimainary assumption - sort the points by x location before computing angle beta
df_byx = df.sort_values(by = ['x'])
df_byx = df_byx.reset_index()
df_byx['beta'] = 0.0

#calculation of beta angle from paper here
#@InProceedings{bogoslavskyi16iros,
# title     = {Fast Range Image-Based Segmentation of Sparse 3D Laser Scans for Online Operation},
# author    = {I. Bogoslavskyi and C. Stachniss},
# booktitle = {Proc. of The International Conference on Intelligent Robots and Systems (IROS)},
# year      = {2016},
# url       = {http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16iros.pdf}
# }
def specialangle(pt1,pt2):
    psi = np.abs(pt1.ang - pt2.ang)
    num = pt1.dist * np.sin(psi)
    den = (pt1.dist - pt2.dist) * np.cos(psi)
    beta = np.arctan2(num,den)
    return beta

for row in range(0,len(df_byx)-1,1):
    df_byx.at[row,'beta'] = specialangle(df_byx.iloc[row], df_byx.iloc[row+1])
# %%


df_byx.to_csv(output)