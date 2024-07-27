import numpy as np
from scipy.spatial import distance

def fit_points_to_circle(points):
    # Calculate the mean of the points
    mean_point = np.mean(points, axis=0)

    # Calculate the distances from the mean point to each point
    distances = [distance.euclidean(point, mean_point) for point in points]

    # Calculate the radius by taking the maximum distance
    radius = max(distances)

    # Return the circle parameters: center (mean point) and radius
    return {'center': mean_point, 'radius': radius}

# Test if the function works correctly
circle_params = fit_points_to_circle([(1, 2), (3, 4), (5, 6)])
print(circle_params)
print('Test passed.' if circle_params['radius'] == 2.23606797749979 else 'Test failed.')
def test_fit_2():
    points = [(0, 0), (1, 1), (2, 2)]
circle_params = fit_points_to_circle(points)
    return circle_params['radius'] == np.sqrt(2) and circle_params['center'][0] == 1 and circle_params['center'][1] == 1
print("___test result: " + str(test_fit_2()) + "___")