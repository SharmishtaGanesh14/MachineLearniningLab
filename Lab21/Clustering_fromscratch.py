import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

def ComputeCentroids(data,labels,k):
    centroids=[]
    for i in range(k):
        datapoints=data[labels==i]
        if len(datapoints) == 0:
            centroid = np.zeros(data.shape[1])  # handle empty cluster
        else:
            centroid=datapoints.mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def assignvalues(data,centroids):
    labels_list=[]
    for x in data:
        distances=[]
        for c in centroids:
            dist=np.linalg.norm(x-c)
            distances.append(dist)
        labels_list.append(np.argmin(distances))
    return np.array(labels_list)

np.random.seed(40)
data=np.array([
    [1, 4],
    [1, 3],
    [0, 4],
    [5, 1],
    [6, 2],
    [4, 0]
])

labels=np.random.choice([0,1],size=data.shape[0])
print("Initial Random Cluster Labels:", labels)
k=2
it=0
while True:
    print(f"ITERATION {it+1}:")
    cen=ComputeCentroids(data,labels,k)
    print("Centroids:\n", cen)
    y=assignvalues(data,cen)
    print("New Labels:", y)
    if np.array_equal(labels, y):
        break
    labels = y
    it += 1

def main():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    X1 = data[:,0]
    X2 = data[:,1]
    ax[0].scatter(X1, X2, color="black")
    for i, (x1, x2) in enumerate(data):
        ax[0].text(x1+0.1, x2, f'{i+1}')
    ax[0].set_title('Initial observations')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[1].scatter(X1, X2, c=labels)
    for i, (x1, x2) in enumerate(data):
        ax[1].text(x1+0.1, x2, f'{i+1}')
    ax[1].set_title(f'After {it+1}nd iteration')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()