import numpy as np
import sys
# You can multiply two lists in numpy #
try:
    a = [1, 2, 3]
    b = [3, 2, 1]
    c = a * b
    print(c)
except:
    print("You can't multiply normal lists")

try:
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])
    c = a * b
    print("But you can do it with numpy!")
    print(c)
except:
    print("You can do that with numpy lists")
# How to get a dimension of the array #
c = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
print(f"Array c {c} has {c.ndim} dimensions.")

# How to gat a shape of the array It's a vector like [1, 3]

print(f"The shape of the c array is {c.shape}")

# What is a size or a type of the array #

print(f"The type of the c array is {c.dtype}")
print(f"And the size of this array is {c.itemsize}")
print(f"The total size of the bites in the array is {c.nbytes}")

# Changing arrays #

d = np.array([[1, 43, 53, 12, 43, 53], [32, 43, 12, 76, 54, 34]])
print(d)
print(f"The first element of this array is {d[0, 0]}")
print(f"The secon collumn of this array is {d[1, :]}")
print(f"The second columns is {d[:, 1]}")
d[0, 0] = 10 # Changing a element of the array #
print(a)

# Initialazing an array #

e = np.zeros((1, 2))
f = np.full((2, 2), 88)
g = np.full_like(a, 31) # Creates a array with the shape of some array #
print(f"This is the initialized array {e}")
print(f"This array is made entirely of 88ths {f}")

# Random numbres array #

g = np.random.rand(4, 2)
h = np.random.randint(1, 9, size=(3, 3))
print(f"This array is made of random numbers {g}")
print(f"This array is made of random numbers {h}")

# Maths in numpy #

i = np.array([1, 2, 3])
j = np.array([4, 5, 6])
print(f"Those are the starting arrays:\n {i}\n{j}")
print(f"\ni + j = {i + j}")
print(f"\ni - j = {i - j}")
print(f"\ni * j = {i * j}")
print(f"\ni / j = {i / j}")
print(f"\ni ^ j = {i ^ j}")

print(f"\ncos(i) = {np.cos(i)}")
print(f"\nsin(i) = {np.sin(i)}")
print(f"\ncos(j) = {np.cos(j)}")
print(f"\nsin(j) = {np.sin(j)}")

# Linear algebra #

k = np.random.randint(1, 9, size=(3, 3))
l= np.random.randint(1, 9, size=(3, 3))

print(f"Matrix one: {k}")
print(f"Matrix one: {l}\n")

l_k = np.matmul(k, l)
print(f"THis to matrices multiplied are:\n {l_k}")

# Statistics #

stats = np.array([[1, 2, 3], [4, 5, 6]])
print(f"This is array stats:\n{stats}")
min_s = np.min(stats)
max_s = np.max(stats)
mean_s = np.mean(stats)
sum_s = np.sum(stats)
print(f"This is max in stats: {max_s}")
print(f"This is min in stats: {min_s}")
print(f"This is mean in stats: {mean_s}")
print(f"This is sum of stats: {sum_s}")







