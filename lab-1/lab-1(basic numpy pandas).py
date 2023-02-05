import numpy as np
import pandas as pd
from timeit import default_timer as timer

# load the dataset

df = pd.read_csv('D:/pattern detection/NIFTY22JANFUT.csv')
df.head()
# datset describe stats

df.describe()
# 4x4 matrix

matrix = np.matrix(df)
matrix2 = matrix[0:4, 2:6]
print(matrix2)

# determinant
start =  timer()
matrix2 = matrix2.astype(np.float64)
determinant =  np.linalg.det(matrix2)
determinant = abs(determinant)
print("\nDeterminant of given 4x4 square matrix:")
print(determinant)
end = timer()

print(end -start)



