# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import namedtuple
from pathlib import Path
import os
Cluster = namedtuple('Point', 'x y z')#meant to convey x,y,z mean of cluster
#repopath = Path(__file__).parent
# fname = r'D:\ObstacleSegmentation\1636310276 - 4-3877 ms.txt'
fname = r'C:\Users\jdroc\OneDrive\Documents\Fall 2021\ROB590\ObstacleSegmentation\1636310276 - 4-3877 ms.txt'
#file = os.path.join(repopath,fname)
#output = os.path.join(repopath,'beta_angles_' + fname)

#create pandas dataframe from asc file
df = pd.read_csv(fname,header=None,delimiter = ',')

#rename the columns based on output from cloudcompare
#df.columns =['x','y','z', 'dist','ang','time','r','g','b','empty']
df.columns = ['x', 'y', 'z', 'r', 'g', 'b', 'dist','ang', 'time']
#df = pd.read_csv('D:/ObstacleSegmentation/1636310276.7642572_LidarSensor1_pointcloud.asc')
npdf = df[['x','y','z']].to_numpy()


# %%
clustering = DBSCAN(eps=.5, min_samples=2).fit(npdf)


# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
plt.title('number of points per cluster')
plt.hist(clustering.labels_)


# %%
df['clusterfromdbscan'] = clustering.labels_
dfsample = df[df.clusterfromdbscan < 10]


# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.set_zlabel('Z global coordinate (m)')
p = ax3D.scatter(dfsample.x,dfsample.y,dfsample.z,c = dfsample.clusterfromdbscan,cmap = plt.get_cmap("plasma"))
# plt.suptitle("Two Rooftop Towers Point Cloud")
plt.title("Two Rooftop Towers Point Cloud\nColored by DBSCAN cluster")
plt.xlabel('X global coordinate (m)')
plt.ylabel('Y global coordinate (m)')
plt.tight_layout()
fig.colorbar(p,ax=ax3D)


# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.set_zlabel('Z global coordinate (m)')
p = ax3D.scatter(df.x,df.y,df.z,c = clustering.labels_,cmap = plt.get_cmap("plasma"))
# plt.suptitle("Two Rooftop Towers Point Cloud")
plt.title("Two Rooftop Towers Point Cloud\nColored by squared Euclidean Distance to nearest neighbor")
plt.xlabel('X global coordinate (m)')
plt.ylabel('Y global coordinate (m)')
plt.tight_layout()
fig.colorbar(p,ax=ax3D)


# %%
df.head()


# %%
exampleclusterdf = df[df['clusterfromdbscan']==0]
exampleclusterdf.describe()


# %%
def XYZmean(df):
    print(type(df))
    xmean=np.mean(df['x'])
    ymean=np.mean(df.y)
    zmean=np.mean(df.z)
    xstd=np.std(df.x)
    ystd=np.std(df.y)
    zstd=np.std(df.z)
    rad=2*max([xstd,ystd,zstd])
    return Cluster(xmean, ymean, zmean, rad) #assigning to tuple "cluster"

cluster_centers=dict.fromkeys(df.clusterfromdbscan.unique())
for cluster in df.clusterfromdbscan.unique():
    tempclusterseries = df[df['clusterfromdbscan']==cluster][['x', 'y', 'z']].agg('mean') #grabs all points associated with cluster in for loop step

    cluster_centers[cluster]= Cluster(tempclusterseries.x,tempclusterseries.y,tempclusterseries.z) #makes cluster series type into dict, 'cluster_centers.x,...'
#print(type(cluster_centers))
#cluster_centers


# %%
#Associate clusters by neighborhood
# np.array(cluster_centers.items())
cluster_center_array=pd.DataFrame.from_dict(cluster_centers,orient='index').to_numpy()
LIDAR_range=100
elevation_angle=(30*np.pi)/180
number_beams=16
raster_buffer=1.25 #25%
distance_between_raster_lines=LIDAR_range*np.tan(elevation_angle / number_beams)*raster_buffer
print(distance_between_raster_lines)
neighborhood_clustering = DBSCAN(eps=distance_between_raster_lines, min_samples=2).fit(cluster_center_array)
neighborhood_clustering_variable = neighborhood_clustering.labels_

# get_ipython().run_line_magic('matplotlib', 'widget')
plt.title('number of points per cluster')
plt.hist(neighborhood_clustering.labels_)


# %%
# for point,newcluster in neighborhood_clustering.labels_
newclusters = pd.DataFrame(neighborhood_clustering.labels_)
newclusters.columns = ['newcluster']
clustercenterdf = pd.DataFrame(cluster_center_array)

clustercenterdf = clustercenterdf.join(newclusters)

clustercenterdf.columns=['x','y','z','newcluster']
print(clustercenterdf)

old2newclustermap = clustercenterdf.newcluster.to_dict()

df['newcluster'] = df['clusterfromdbscan'].map(old2newclustermap)


# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.set_zlabel('Z global coordinate (m)')
#p = ax3D.scatter(df.x,df.y,df.z)#original points colored by neighborhood, removed "",c = df.newcluster,cmap = plt.get_cmap("plasma")""
p = ax3D.scatter(clustercenterdf.x,clustercenterdf.y,clustercenterdf.z,c = clustercenterdf.newcluster,cmap = plt.get_cmap("inferno"))#just centers that were basis for neighbor
# plt.suptitle("Two Rooftop Towers Point Cloud")
plt.title("Two Rooftop Towers Point Cloud\nColored by squared Euclidean Distance to nearest neighbor")
plt.xlabel('X global coordinate (m)')
plt.ylabel('Y global coordinate (m)')
plt.tight_layout()
fig.colorbar(p,ax=ax3D)


# %%
exampleclusterdf2 = df[df['clusterfromdbscan']==1]
exampleclusterdf2.describe()


# %%
def specialangle(pt1,pt2):
    psi = np.abs(pt1.ang - pt2.ang)
    num = pt1.dist * np.sin(psi)
    den = (pt1.dist - pt2.dist) * np.cos(psi)
    beta = np.arctan2(num,den)
    return beta

for cluster in clustercenterdf.newcluster.unique(): #loop through all points in cluster neighborhood
    for point1 in clustercenterdf[clustercenterdf.newcluster==cluster].iterrows():
        print(point1)
        #for point2 in clustercenterdf[clustercenterdf.newcluster==cluster].iterrows():
		#   If point1 = point2
		#		continue
		#	Else

#for row in range(0,len(df_byx)-1,1):
#    df_byx.at[row,'beta'] = specialangle(df_byx.iloc[row], df_byx.iloc[row+1])



# %%
def calculate_center(thiscluster): #only consider x and y for intersection overlap algorithm, downstream from neighborhood clusters
    x_center = thiscluster.x.mean()
    y_center = thiscluster.y.mean()
    x_std = thiscluster.x.std()
    y_std = thiscluster.y.std()
    rad = 2*max([x_std,y_std])
    return x_center,y_center,rad


# %%
circ1 = calculate_center(exampleclusterdf)
circ2 = calculate_center(exampleclusterdf2)
print(f"cluster1 {circ1}\ncluster2 {circ2}")


# %%
cluster_radius=dict.fromkeys(df.clusterfromdbscan.unique())
for cluster in df.clusterfromdbscan.unique():
    [x_center, y_center,rad] = calculate_center(df[df['clusterfromdbscan']==cluster]) #grabs all points associated with cluster in for loop step

    cluster_radius[cluster]= [x_center, y_center,rad]


# %%
cluster_radius_array=pd.DataFrame.from_dict(cluster_radius,orient='index')
cluster_radius_array.columns=['x','y','rad']
cluster_radius_array


# %%
def intersection_area(x0, y0, R, x1, y1, r):
    """Return the area of intersection of two circles.

    The circles have radii R and r and centers (x1,y1) and (x2,y2).

    """
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2 / (np.pi * min(R,r)**2)
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta -
             0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
           ) / (np.pi * min(R,r)**2) #area of intersection normalized by area of smallest circle, % containment
intersection_area(circ1[0],circ1[1],circ1[2],circ2[0],circ2[1],circ2[2])


# %%
#return overlap score for each pair of clusters
ptlist = cluster_radius_array.index.to_list()
overlapsdict = dict.fromkeys(ptlist)
for idx1,pt1 in cluster_radius_array.iterrows():
    overlapsdict[idx1] = dict.fromkeys(ptlist)
    for idx2,pt2 in cluster_radius_array.iterrows():
        overlapsdict[idx1][idx2] = intersection_area(pt1.x, pt1.y, pt1.rad, pt2.x, pt2.y, pt2.rad)


# %%
pd.DataFrame(overlapsdict)


# %%
#create dictionary of points with threshold of each root point
overlap_thresh=0.75
closepointsdict = dict.fromkeys(ptlist)
for point in overlapsdict.keys():
    ptsinrange = []
    for point2, distance in overlapsdict[point].items():
        if distance > overlap_thresh:
            ptsinrange.append(point2)
    closepointsdict[point] = ptsinrange


# %%
closepointsdict[5]


# %%
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


# %%
adjacent_confirmed = set()
ungroupedpts = ptlist
need2checkstill = set()
neighbors = getadjacent(adjacent_confirmed,ungroupedpts[3],need2checkstill)
neighbors


# %%
groupnum = 0
ungroupedpts = ptlist
print(ungroupedpts)
clusters_dict = {}
while len(ungroupedpts) > 0:
    adjacent_confirmed = set()
    need2checkstill = set()
    neighbors = getadjacent(adjacent_confirmed,ungroupedpts[0],need2checkstill)

    clusters_dict[groupnum] = neighbors

    if len(neighbors)==0:
        neighbors = [ungroupedpts[0]]
    print(neighbors)
    ungroupedpts = list(set(ungroupedpts) - set(neighbors))
    groupnum+=1


# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
plt.bar(clusters_dict.keys(),[len(v) for k,v in clusters_dict.items()])


# %%
