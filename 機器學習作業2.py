import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# 群集中心和元素的數量
seed_num = 2
dot_num = 6

# 初始元素
x=np.array([1,1,0,5,6,4])
y=np.array([4,3,4,1,2,0])
df=pd.DataFrame({'X1':x,'X2':y})
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['X1'], df['X2'], color = 'red')
ax.set_title('data')
plt.show()
# 初始群集中心
kx = np.random.randint(min(x), max(x)+1, seed_num)
ky = np.random.randint(min(y), max(y)+1, seed_num)


# 兩點之間的距離
def dis(x, y, kx, ky):
    return (((kx - x) ** 2 + (ky - y) ** 2) ** 0.5)


# 對每筆元素進行分群
def cluster(x, y, kx, ky):
    team = []
    x_team=[]
    y_team=[]
    for i in range(2):
        team.append([])
        x_team.append([])
        y_team.append([])

    mid_dis = 99999999
    for i in range(dot_num):
        for j in range(seed_num):
            distant = dis(x[i], y[i], kx[j], ky[j])
            if distant < mid_dis:
                mid_dis = distant
                flag = j
        team[flag].append([x[i], y[i]])
        x_team[flag].append(x[i])
        y_team[flag].append(y[i])
        mid_dis = 99999999
    return team,x_team,y_team


# 對分群完的元素找出新的群集中心
def re_seed(team, kx, ky):
    sumx = 0
    sumy = 0
    new_seed = []
    for index, nodes in enumerate(team):
        if nodes == []:
            new_seed.append([kx[index], ky[index]])#可加可不加
        for node in nodes:
            sumx += node[0]
            sumy += node[1]
        new_seed.append([(sumx / len(nodes)), (sumy / len(nodes))])
        sumx = 0
        sumy = 0
    nkx = []
    nky = []
    for i in new_seed:
        nkx.append(i[0])
        nky.append(i[1])
    return nkx, nky,new_seed


# k-means 分群
def kmeans(x, y, kx, ky, fig):
    team,x_team,y_team = cluster(x, y, kx, ky)
    nkx, nky,new_seed = re_seed(team, kx, ky)
    print(team)
    print(kx)
    print(ky)
    print(new_seed)
    print(nkx)
    print(nky)
    # plot: nodes connect to seeds
    cx = []
    cy = []
    line = plt.gca()
    for index, nodes in enumerate(team):
        for node in nodes:
            cx.append([node[0], nkx[index]])
            cy.append([node[1], nky[index]])
        for i in range(len(cx)):
            line.plot(cx[i], cy[i], color='r', alpha=0.6)
        cx = []
        cy = []



    feature1 = plt.scatter(x_team[0], y_team[0], c='b')
    feature2 = plt.scatter(x_team[1], y_team[1], c='m')


    k_feature = plt.scatter(kx, ky,c='g')
    nk_feaure = plt.scatter(np.array(nkx), np.array(nky),c='y', s=50)
    plt.show()

    # 判斷群集中心是否不再更動
    if nkx == list(kx) and nky == list(ky):
        return
    else:
        fig += 1
        kmeans(x, y, nkx, nky, fig)


kmeans(x, y, kx, ky, fig=0)
