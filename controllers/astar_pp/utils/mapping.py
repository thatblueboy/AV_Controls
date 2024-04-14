

class findObs:
    def __init__(self, lidar):
        self.lidar = lidar
        self.inf = 200
        self.min_dist = 0.0
        self.min_dist_index = 0
        self.min_dist_point = [0, 0, 0]
        self.derivative_threshold = 10
        self.min_width = 0

    def computeDerivative(self, scan):
        jumps = [0]
        for i in range(1, len(scan) - 1):
            l = scan[i - 1]
            r = scan[i + 1]
            if l >= self.min_dist and r >= self.min_dist:
                derivative = (r - l) / 2
                jumps.append(derivative)
            else:
                jumps.append(0)
        jumps.append(0)
        return jumps
    
    def processLidarData(self, scanData):
        # consider infinity as 50
        scanData = [self.inf if x == float('inf') else x for x in scanData]
        return scanData
    
    def getCylinders(self, scanData):
        # consider infinity as 50
        scanData = self.processLidarData(scanData)
        derivative = self.computeDerivative(scanData)

        cylinder_list = []
        on_cylinder = False
        sum_ray, sum_depth, rays = 0.0, 0.0, 0

        for i in range(len(derivative)):
            # Check if the derivative indicates a significant change in depth
            if abs(derivative[i]) > self.derivative_threshold:
                if not on_cylinder:
                    on_cylinder = True
                    sum_ray, sum_depth, rays = 0.0, 0.0, 0

                sum_ray += i
                sum_depth += scanData[i]
                rays += 1
            else:
                if on_cylinder:
                    # Check if enough points were detected to consider it a cylinder
                    if rays > self.min_width:
                        # Calculate average ray and depth for the cylinder
                        average_ray = sum_ray / rays
                        average_depth = sum_depth / rays
                        cylinder_list.append((average_ray, average_depth))

                    on_cylinder = False

        return cylinder_list
        
