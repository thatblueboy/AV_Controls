## Planning and Control Pipelines for Autonomous Vehicles
Repository for implementations of various motion planning and control pipelines for autonomous vehicles.




### Controllers

1. ``astar_pp``: Uses Astar algorithm to plan in a local frame and pure pursuit to track local path in a one step sense.

2. ``frenet``: Based on optimal trajectory planning in a frenet frame[1]. MPC is to be used to track the given reference trajectory.

### Reference
[1]https://www.researchgate.net/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame

### Code Reference

1. https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/FrenetOptimalTrajectory

