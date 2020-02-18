import numpy as np
import csv, argparse, sys
sys.path.append('../cameramouse')
from interface import WindowsMonitor
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fname", "--filename", type=str, required=True,
	help="filename of data you want to test: still_fist, circle, up_down, left_right, finger_extensions")
ap.add_argument("-os", "--os", type=str, required=False, default="win",
	help="your OS: win, linux")
ap.add_argument("-filt", "--filter", type=str, required=False,
	help="filter you want to test: averaging ")
ap.add_argument("-p", "--plot", type=bool, default=True,
	help="whether you want to plot or not")
args = ap.parse_args()

class Filter():
    def __init__(self, rects, centroids, vels, areas, times):
        self.rects = rects
        self.centroids = centroids
        self.vels = vels
        self.areas = areas
        self.times = times
        self.cursor_initial = [monitor.width//2, monitor.height//2]

    def filter(self):
        pass

    def map_vel_to_pixels(self, vel):
        # tf = tanh((1/10)x - 2) + 1 where x is the speed in terms of movement on the screen
        g_x = (np.tanh(1/10*(vel[0]/cam_res[0])-2) + 1) # hyperbolic function gain can be between 0 and 2
        g_y = (np.tanh(1/10*(vel[1]/cam_res[1])-2) + 1)
        print("{}, {}".format(g_x, g_y))
        ret_x = int(vel[0] * g_x)
        ret_y = int(vel[1] * g_y)

        return [ret_x, ret_y]

    def simulate_cursor_unfilt(self):
        cursor_positions = []
        cursor_positions.append(self.cursor_initial)
        for i in range(1, len(self.centroids)):
            dt = self.times[i] - self.times[i-1]
            dx = self.centroids[i][0] - self.centroids[i-1][0]
            dy = self.centroids[i][1] - self.centroids[i-1][1]
            v = self.map_vel_to_pixels([-dx/dt, -dy/dt])
            pos = [v[0] + cursor_positions[i-1][0], v[1] + cursor_positions[i-1][1]]
            if pos[0] > monitor.width:
                pos[0] = monitor.width
            elif pos[0] < 0:
                pos[0] = 0
            if pos[1] > monitor.height:
                pos[1] = monitor.height
            elif pos[1] < 0:
                pos[1] = 0
            cursor_positions.append(pos)
        return np.array(cursor_positions)

    def simulate_cursor(self):
        self.filter()
        cursor_positions = []
        cursor_positions.append(self.cursor_initial)
        for i in range(1, len(self.averaged_positions)):
            dt = self.times[i] - self.times[i-1]
            dx = self.averaged_positions[i][0] - self.averaged_positions[i-1][0]
            dy = self.averaged_positions[i][1] - self.averaged_positions[i-1][1]
            v = self.map_vel_to_pixels([-dx/dt, -dy/dt])
            pos = [v[0] + cursor_positions[i-1][0], v[1] + cursor_positions[i-1][1]]
            if pos[0] > monitor.width:
                pos[0] = monitor.width
            elif pos[0] < 0:
                pos[0] = 0
            if pos[1] > monitor.height:
                pos[1] = monitor.height
            elif pos[1] < 0:
                pos[1] = 0
            cursor_positions.append(pos)

        return np.array(cursor_positions)

class AveragingFilter(Filter):
    def __init__(self,rects, centroids, vels, areas, times, filter_size):
        super().__init__(rects, centroids, vels, areas, times)
        self.filter_size = filter_size

    def filter(self):
        self.averaged_positions = []
        for i in range(0, len(self.centroids)-self.filter_size):
            x = np.average(self.centroids[i:i+self.filter_size, 0])
            y = np.average(self.centroids[i:i+self.filter_size, 1])
            self.averaged_positions.append([x, y])


raw_data = np.loadtxt(args.filename+'.csv', delimiter=',')
rectangles = raw_data[:, 0:4]
centroids = raw_data[:, 4:6]
velocities = raw_data[:, 6:8]
areas = raw_data[:, 8]
timestamps = raw_data[:, 9] - raw_data[0, 9] # load times and set initial time to zero

if args.os == "win":
    global monitor
    monitor = WindowsMonitor()
elif args.os == "linux":
    pass

#TODO Make this automatic
global cam_res
cam_res = [1280, 720]

if args.filter == "averaging":
    filter = AveragingFilter(rectangles, centroids, velocities, areas, timestamps, 5)
else:
    pass

positions = filter.simulate_cursor()
positions_unfilt = filter.simulate_cursor_unfilt()

if args.plot == True:
    plt.figure()
    plt.plot(positions[1, 0], positions[1, 1], 'o')
    plt.xlim([0, monitor.width])
    plt.ylim([0, monitor.height])
    for i in range(1, len(positions)):
        plt.plot(positions[i, 0], positions[i, 1], 'x')
        plt.plot(positions_unfilt[i+filter.filter_size, 0], positions_unfilt[i+filter.filter_size, 1], 'o')
        plt.pause(0.02)

    plt.show()
