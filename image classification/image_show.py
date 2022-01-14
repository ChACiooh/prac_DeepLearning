import matplotlib.pyplot as plt
import numpy as np

# 각 숫자 라벨별 class

# 0: airplane
# 1: automobile
# 2: bird
# 3: cat
# 4: deer
# 5: dog
# 6: frog
# 7: horse
# 8: ship
# 9: truck

def img_show(i):
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")

    plt.imshow(x_train[i], interpolation="bicubic")
    plt.show()
    print(y_train[i])
    
def img_show_n(n):
    num_train = len(np.load("data/x_train.npy"))
    shuffled_idx = np.arange(0, num_train)
    np.random.shuffle(shuffled_idx)
    start = shuffled_idx[0] if num_train - shuffled_idx[0] > 10 else shuffled_idx[0] - 10
    for i in range(start, start+n):
        img_show(i)