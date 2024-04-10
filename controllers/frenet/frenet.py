from vehicle import Driver
from controller import GPS, Accelerometer
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import bisect
from utils import FrenetPlanner 

# variables
time_step=40
# create the Robot instance.
driver = Driver()

accelerometer = Accelerometer('accelerometer_center') #accelerometer at center of rear axle

gps_c=GPS('gps_center') #GPS at center of rear axle
gps_c.enable(time_step)

gps_t=GPS('gps_top') #GPS at top of car. refer https://cyberbotics.com/doc/automobile/car?version=R2020a#positions-of-the-car-sensor-slots
gps_t.enable(time_step)

waypoints = [ [-850, 6.85], 
             [-870, 6.85],
    [-890, 6.85],
    [-910, 6.85],
    [-930, 6.85],
    [-950, 6.85],
    [-970, 6.85],
    [-990, 6.85],
    [-1010, 6.85],
    [-1030, 6.85],
    [-1050, 6.85],
    [-1070, 6.85],
    [-1090, 6.85],
    [-1110, 6.85]
] # issue: perpendicular from position of car to spline interpolating waypoints should exist

planner = FrenetPlanner(waypoints)
def remove_waypoints_behind_car(waypoints, car_x):
    filtered_waypoints = [point for point in waypoints if point[0] <= car_x]
    return filtered_waypoints


while driver.step() != -1:
    
    s = driver.getCurrentSpeed()
    acc = accelerometer.getValues()
    pos = np.array(gps_c.getValues()) 
    pos_top = np.array(gps_t.getValues()) 

    heading = (pos_top - pos)/np.linalg.norm(pos_top - pos)
    vel = heading[:2] * s

    frenet_path = planner.get_path(pos[:2], vel[:2], acc[:2])
    print(frenet_path.x)
    
    driver.setCruisingSpeed(5)
    
    print(driver.getCurrentSpeed())