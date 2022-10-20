from turtle import shape
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ax = plt.axes()
def powerIteration(mat, iterations):
    bVector = np.random.rand(mat.shape[1])
    for _ in range(iterations):
        bpdt = np.dot(mat, bVector)
        norm = np.linalg.norm(bpdt)

        bVector = bpdt/norm

        ax.plot((0, bVector[0]), (0, bVector[1]))

    return bVector

print("Eigen value through power iteration : ",powerIteration(np.array([[0.5,0.9],[1.2,55]]),20))
preseteig = np.linalg.eig(np.array([[0.5,0.9],[1.2,55]]))
print("Eigne value through numpy : ", preseteig)
plt.show()
