import numpy as np
import os 
x=np.arange(16).reshape(4,4)
header='C1 C2 C3 C4'
np.savetxt('7_array.txt',x,fmt="%d",header=header)
print("\nAftet loading,content of the text file:")
print(np.loadtxt('7_array.txt')) 