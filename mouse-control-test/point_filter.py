import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz

class Data:
    def __init__(self, data):
        self.x = data[1:, 0]                 # read x mouse_data
        self.y = data[1:, 1]                 # read y data
        self.t = data[1:, 2]/1000            # read time and convert to seconds

        self.dt = self.t[1:] - self.t[:-1]             # calculate dt

        self.vx = (self.x[1:] - self.x[:-1])/self.dt       # calculate x velocity
        self.vy = (self.y[1:] - self.y[:-1])/self.dt       # calculate y velocity

        self.x_av = np.average(self.x)         # calulate average x
        self.y_av = np.average(self.y)         # calculate average y
        self.x_std = np.std(self.x)
        self.y_std = np.std(self.y)

'''Grab data from log file'''
data_stat = np.genfromtxt('stay_still.csv', delimiter=',')
data_right = np.genfromtxt('move_right.csv', delimiter=',')

stat = Data(data_stat)
right = Data(data_right)

print("Static Mean X: %f, Static Standard Deviation X: %f" % (stat.x_av, stat.x_std)) # std_x is 0.001204
print("Static Mean Y: %f, Static Standard Deviation Y: %f" % (stat.y_av, stat.y_std)) # std_y is 0.001767

print("Moving Mean X: %f, Moving Standard Deviation X: %f" % (right.x_av, right.x_std))
print("Moving Mean Y: %f, Moving Standard Deviation Y: %f" % (right.y_av, right.y_std))

'''Set filter coefficients'''
n = 20
taps = np.ones(n)/n

'''Apply the filter to both sets of data'''
stat.filt_x = lfilter(taps, 1.0, stat.x)
stat.filt_y = lfilter(taps, 1.0, stat.y)
right.filt_x = lfilter(taps, 1.0, right.x)
right.filt_y = lfilter(taps, 1.0, right.y)

'''Setup a 2 by 2 plot for showing data'''
fig, axs = plt.subplots(nrows=2, ncols=2, sharex='row', sharey='row')
axs[1,0].set_title('Original Moving Data')
axs[1,1].set_title('Filtered Moving Data')
axs[0,0].set_title('Original Static Data')
axs[0,1].set_title('Filtered Static Data')

'''Plot the true path to compare our results to'''
axs[0,0].plot(stat.x_av, stat.y_av, 'x')
axs[1,0].plot(right.x, np.ones(len(right.y))*right.y_av)
axs[0,1].plot(stat.x_av, stat.y_av, 'x')
axs[1,1].plot(right.x, np.ones(len(right.y))*right.y_av)

'''Plot points at two times the real speed'''
for i in range(n, len(right.filt_x)-1):
    axs[1,0].scatter(right.x[i], right.y[i])
    axs[1,1].scatter(right.filt_x[i], right.filt_y[i])
    axs[0,0].scatter(stat.x[i], stat.y[i])
    axs[0,1].scatter(stat.filt_x[i], stat.filt_y[i])
    plt.pause(right.dt[i]/2)

plt.show()
