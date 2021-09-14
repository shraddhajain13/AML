import numpy as np

list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
list3 = []
list3.append(list1)
list3.append(list2)
print(list3)

np.savetxt(r"D:\Shraddha\temp.csv", np.array(list3).transpose(), delimiter = ',')