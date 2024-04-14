from math import sin,atan2,hypot
import numpy as np

class PurePursuit():
    def __init__(self, path, lookahead, velocity, current=[0,0]):
        self.path = path
        self.current = current
        self.target=[0,0]
        self.lookahead = lookahead
        self.velocity = velocity
        self.nearest=0
        self.WB=3.5
    #calculates distance between 2 points and inclination of resulting line
    def getParams(self,point1, point2):
       x_diff = point2[0] - point1[0]
       y_diff = point2[1] - point1[1]
       angle =atan2(y_diff, x_diff)
       dist=hypot(y_diff,x_diff)
       return [dist, angle]

    #updates target point using look_ahead distance
    def setTarget(self):
        try:
            self.nearest = self.indexOfClosestPoint(self.path, 0, self.current)
            self.path=self.path[self.nearest:]
            targetIndex = self.indexOfClosestPoint(self.path[self.nearest:], self.lookahead, self.current)
            self.target=self.path[targetIndex]
            print(self.current[0],self.current[1])

        except IndexError:
            self.velocity=0
            self.WB=0

    @staticmethod
    def indexOfClosestPoint(points, radius, target_point):
        #find index of point in list with least abs(distance to target point- radius)
        
        index_least = 0
        min_difference = abs(hypot(points[0][0] - target_point[0], points[0][1] - target_point[1]) - radius)

        for i in range(1, len(points)):
            dist = hypot(points[i][0] - target_point[0], points[i][1] - target_point[1])
            current_difference = abs(dist - radius)

            if current_difference < min_difference:
                min_difference = current_difference
                index_least = i

        return index_least
        
    def steeringAngle(self, yaw):
        # print(self.path)
        self.setTarget()
        target_yaw = self.getParams(self.current, self.target)[1]
        print(target_yaw)
        steer = np.arctan2(self.WB*2*sin(target_yaw - yaw)/self.lookahead,1.0)
        return steer
    
    def re(self, path):
        self.path = path