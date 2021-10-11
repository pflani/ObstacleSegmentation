import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import sys
import json
import math

repopath = Path(__file__).parent
fname = r'1632344375 - two towers and noise.asc'
file = os.path.join(repopath,fname)
output = os.path.join(repopath,'beta_angles_' + fname)

#create pandas dataframe from asc file
df = pd.read_csv(file,header=None,delimiter = ' ')

#rename the columns based on output from cloudcompare 
# df.columns =['x','y','z', 'dist','ang','time','r','g','b']
df.columns = ['x', 'y', 'z', 'dist', 'r', 'g', 'b','ang', 'time']

# if we want to try matrix multiplication on a machine with a GPU we may realize improvements using CuPy
# diffmat=np.diag(-1*np.ones(len(df)-1),1)
# diffmat[:,1] = 1
# xdiffs=diffmat*df.x

#select a subset of points to keep things smaller
# smallsetdf = df[:1000]

smallsetdf = df
# %%
#prepolulate a dictionary with keys as the list of points we are comparing distances between
ptlist = smallsetdf.index.to_list()
allthedistsdict = dict.fromkeys(ptlist)


# %%
#plot the histogram of point distances
smallsetdf.x.diff().hist()

dist_thresh = .2

print('Exhaustively compute the Euclidean Distance betwen points (brute force)\n')

#calculate the difference between point a and point b, adding to the dictionary
for idx1,pt1 in tqdm(smallsetdf.iterrows(),total=len(smallsetdf)):
    allthedistsdict[idx1] = dict.fromkeys(ptlist)
    for idx2,pt2 in smallsetdf.iterrows():
        #ignore the square root to save some compute instructions
        allthedistsdict[idx1][idx2] = math.pow(pt1.x - pt2.x,2) + math.pow(pt1.y-pt2.y,2) + math.pow(pt1.z-pt2.z,2)

print('Saving distance results to csv\n')
distsdf = pd.DataFrame(allthedistsdict)
distsdf.to_csv(os.path.join(repopath,'euclideandistancebetweeneverycombination.csv'))


print('Create dictionary of points with threshold of each root point\n')
closepointsdict = dict.fromkeys(ptlist)
for point in tqdm(allthedistsdict.keys(), total= len(allthedistsdict)):
    ptsinrange = []
    for point2, distance in allthedistsdict[point].items():
        if distance < dist_thresh:
            ptsinrange.append(point2)
    closepointsdict[point] = ptsinrange

# with open(r"/Users/jdrockton/OneDrive/Documents/Fall 2021/ROB590/ObstacleSegmentation/closepointsdict_.2thres.json", "w") as outfile:
#     json.dump(closepointsdict, outfile)



def getadjacent(confirmedhits,pt2check,need2checkstill):
    # current code requires set to be closed (e.g. dictionary must not reference points that aren't in dictionary
    # if pt2check in testdict: 
    need2checkstill = need2checkstill | (set(closepointsdict[pt2check]) - confirmedhits)
    if len(need2checkstill) > 0:
        thispt = need2checkstill.pop()
        confirmedhits = confirmedhits | set([thispt])
        return getadjacent(confirmedhits,thispt,need2checkstill)
    else:
        return confirmedhits


#Call getadjacent function to compute groups until all points belong to a group

groupnum = 0
ungroupedpts = ptlist
clusters_dict = {}
while len(ungroupedpts) > 0:
    adjacent_confirmed = set()
    need2checkstill = set()
    neighbors = getadjacent(adjacent_confirmed,ungroupedpts[0],need2checkstill)
    clusters_dict[groupnum] = neighbors
    ungroupedpts = list(set(ungroupedpts) - set(neighbors))
    groupnum+=1

plt.bar(clusters_dict.keys(),[len(v) for k,v in clusters_dict.items()])

# %%
clustersdataframe = pd.DataFrame.from_dict(clusters_dict,orient = 'index')

clustersdataframe.to_csv(os.path.join(repopath,'euclideanclusters_distthres_'+str(dist_thresh)+'.csv'))

print('Identified ' + str(len(clusters_dict.keys())) + ' clusters!\n')


#if you want a 3d scatter plot of the grouped points, colored by cluster
# %matplotlib widget
# fig = plt.figure()
# ax3D = fig.add_subplot(111, projection='3d')
# ax3D.scatter(df.x,df.y,df.z,c = df['clusternum'],cmap = plt.get_cmap("plasma"))


reversedclusterdict = {}
for k,v in clusters_dict.items():
    for val in v:
        reversedclusterdict[val] = k


def secondmin(x):
    return x.where(~(x == x.min())).min()
# df['dist2closest'] = df2.apply(secondmin,axis = 1)

print('Re-exporting point cloud with cluster number appended\n')
df['clusternum'] = df.index.map(reversedclusterdict)
df.to_csv(os.path.join(repopath,'clustered_'+fname + '.csv'))
# import numpy as np

# diffmat=np.diag(-1*np.ones(len(df)-1),1)
# diffmat[:,1] = 1
# xdiffs=diffmat*df.x









# df2 = pd.read_csv(r'/Users/jdrockton/OneDrive/Documents/Fall 2021/ROB590/ObstacleSegmentation/euclideandistancebetweeneverycombination.csv')
# df2.drop(df2.columns[0],axis=1,inplace=True)


# %%
# %matplotlib widget
# fig = plt.figure()
# ax3D = fig.add_subplot(111, projection='3d')
# ax3D.set_zlabel('Z global coordinate (m)')
# p = ax3D.scatter(df.x,df.y,df.z,c = df['dist2closest'],cmap = plt.get_cmap("plasma"))
# # plt.suptitle("Two Rooftop Towers Point Cloud")
# plt.title("Two Rooftop Towers Point Cloud\nColored by squared Euclidean Distance to nearest neighbor")
# plt.xlabel('X global coordinate (m)')
# plt.ylabel('Y global coordinate (m)')
# plt.tight_layout()
# fig.colorbar(p,ax=ax3D)