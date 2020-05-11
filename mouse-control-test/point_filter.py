import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz, butter, lfilter_zi, filtfilt

class Data:
    def __init__(self, data):
        self.x = data[1:, 0]*1280                      # read x mouse_data
        self.y = data[1:, 1]*780                       # read y data
        self.t = data[1:, 2]/1000                      # read time and convert to seconds

        self.xn = self.x + np.random.normal(0, 1.5, len(self.x))
        self.yn = self.y + np.random.normal(0, 1.5, len(self.x))

        self.dt = self.t[1:] - self.t[:-1]             # calculate dt

        self.vx = (self.x[1:] - self.x[:-1])/self.dt   # calculate x velocity
        self.vy = (self.y[1:] - self.y[:-1])/self.dt   # calculate y velocity

        self.x_av = np.average(self.x)                 # calulate average x
        self.y_av = np.average(self.y)                 # calculate average y
        self.x_std = np.std(self.x)
        self.y_std = np.std(self.y)

'''Grab data from log file'''
data_stat = np.genfromtxt('mouse_data/stay_still.csv', delimiter=',')
data_right = np.genfromtxt('mouse_data/move_right.csv', delimiter=',')
data_circle = np.genfromtxt('mouse_data/circle.csv', delimiter=',')

stat = Data(data_stat)
right = Data(data_right)
circle = Data(data_circle)
circle.vx = (circle.xn[1:] - circle.xn[:-1])/circle.dt   # calculate x velocity
circle.vy = (circle.yn[1:] - circle.yn[:-1])/circle.dt   # calculate y velocity


print("Static Mean X: %f, Static Standard Deviation X: %f" % (stat.x_av, stat.x_std)) # std_x is 1.5
print("Static Mean Y: %f, Static Standard Deviation Y: %f" % (stat.y_av, stat.y_std)) # std_y is 1.5

print("Moving Mean X: %f, Moving Standard Deviation X: %f" % (right.x_av, right.x_std))
print("Moving Mean Y: %f, Moving Standard Deviation Y: %f" % (right.y_av, right.y_std))

print("Circle Mean X: %f, Moving Standard Deviation X: %f" % (circle.x_av, circle.x_std))
print("Circle Mean Y: %f, Moving Standard Deviation Y: %f" % (circle.y_av, circle.y_std))

FIR = 1

'''Apply the filter to both sets of data'''
if FIR:
    n = 10
    taps = np.ones(n)/n
    stat.filt_x = lfilter(taps, 1.0, stat.x)
    stat.filt_y = lfilter(taps, 1.0, stat.y)
    right.filt_x = lfilter(taps, 1.0, right.x)
    right.filt_y = lfilter(taps, 1.0, right.y)
    circle.filt_x = lfilter(taps, 1.0, circle.xn)
    circle.filt_y = lfilter(taps, 1.0, circle.yn)
else:
    b, a = butter(1, 0.05)
    print(b)
    print(a)
    zi = lfilter_zi(b, a)
    n = 1
    stat.filt_x, _ = lfilter(b, a, stat.x, zi=zi*stat.x[0])
    stat.filt_y, _ = lfilter(b, a, stat.y, zi=zi*stat.y[0])
    right.filt_x,  _ = lfilter(b, a, right.x, zi=zi*right.x[0])
    right.filt_y, _ = lfilter(b, a, right.y, zi=zi*right.y[0])
    circle.filt_x, _= lfilter(b, a, circle.xn, zi=zi*circle.xn[0])
    circle.filt_y, _ = lfilter(b, a, circle.yn, zi=zi*circle.yn[0])

'''Setup a 2 by 2 plot for showing data'''
fig, axs = plt.subplots(nrows=3, ncols=2, sharex='row', sharey='row')
axs[1,0].set_title('Original Moving Data')
axs[1,1].set_title('Filtered Moving Data')
axs[0,0].set_title('Original Static Data')
axs[0,1].set_title('Filtered Static Data')
axs[2,0].set_title('Original Circle Data')
axs[2,1].set_title('Filtered Circle Data')


'''Plot the true path to compare our results to'''
axs[0,0].plot(stat.x_av, stat.y_av, 'x')
axs[1,0].plot(right.x, np.ones(len(right.y))*right.y_av)
axs[0,1].plot(stat.x_av, stat.y_av, 'x')
axs[1,1].plot(right.x, np.ones(len(right.y))*right.y_av)
axs[2,0].plot(circle.x, circle.y, 'g')
axs[2,1].plot(circle.x, circle.y, 'g')


'''Plot points at two times the real speed'''
for i in range(n, len(right.filt_x)-1):
    axs[1,0].scatter(right.x[i], right.y[i])
    axs[1,1].scatter(right.filt_x[i], right.filt_y[i])
    axs[0,0].scatter(stat.x[i], stat.y[i])
    axs[0,1].scatter(stat.filt_x[i], stat.filt_y[i])
    axs[2,0].scatter(circle.xn[i], circle.yn[i])
    axs[2,1].scatter(circle.filt_x[i], circle.filt_y[i])
    plt.pause(right.dt[i]/2)

plt.show()
