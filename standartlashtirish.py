# 31.10.2024

# iris ba'zasidagi har bir ustunni standartlashtirish

import numpy as np
import math
# Standartlashtirish usuli
xy = np.loadtxt('iris.csv', delimiter=',', dtype=str)
x = xy[1:, :-1]  # So'nggi ustunni (gulli tur) olib tashlash
x = x.astype('float')
print("Original data:\n", x)

n, m = x.shape  # Qatorlar soni (n) va ustunlar soni (m)
for k in range(m):
    y = x[:, k]
    myu = np.mean(y)  # O'rtacha qiymat
    s = np.std(y)  # Standart chetlanish

    # Standartlashtirish
    for i in range(n):
        x[i, k] = (y[i] - myu) / s

# Matnli shaklga o'tkazish
x = x.astype('str')
print("Standardized data:\n", x)