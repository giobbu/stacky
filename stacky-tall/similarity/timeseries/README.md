# Dynamic Time Warping

## Basic Idea
In time series analysis, dynamic time warping (DTW) is an algorithm for measuring similarity between two temporal sequences X and Y, which may vary in speed [Ref]. The objective of DTW is to temporally align these two sequences in some optimal sense under certain constraints.
* Boundary condition
* Monotonicity condition
* Step-size condition

### Cost Matrix through a Local Cost Measure
![Image Alt Text](img/cost_matrix.png)

### Accumulated Cost Matrix with Dynamic Programming
![Image Alt Text](img/accumulated_cost_matrix.png)

### DTW Distance
![Image Alt Text](img/dtw_distance.png)