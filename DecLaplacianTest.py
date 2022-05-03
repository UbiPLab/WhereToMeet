import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


if __name__=="__main__":
    k=10
    A=np.arange(20)
    print(A)
    idx = np.argpartition(A, -k)[-k:]
    print(idx)