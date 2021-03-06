from numpy import pi, sin, linspace
from araucaria.stats import roll_med
import matplotlib.pyplot as plt
# generating a signal and its rolling median
f1   = 0.2 # frequency
t    = linspace(0,10)
y    = sin(2*pi*f1*t)
line = plt.plot(t,y, label='signal')
for method in ['calc', 'extend', 'nan']:
   fy   = roll_med(y, window=25, edgemethod=method)
   line = plt.plot(t, fy, marker='o', label=method)
lab = plt.xlabel('t')
lab =plt.ylabel('y')
leg = plt.legend()
plt.show(block=False)
