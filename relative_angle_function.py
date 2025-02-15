# create a line between two points
line1 = point1 - point2
line2 = point3 - point2

# the related angle from line 1 to line 2 
# you can watch: https://en.neurochispas.com/physics/direction-of-a-2d-vector-formulas-and-examples/
def calculate_theta(line1, line2):
    m = np.shape(line1)[0]
    theta = np.zeros(m)
    R = np.zeros((m, 2, 2))
    c = np.zeros((m, 2))
    for i in range(m):
        A = line1[i, :]
        B = data2[i, :]
        nor_A = A / np.linalg.norm(A)
        nor_B = B / np.linalg.norm(B)
        R[i] = np.array([[nor_A[0], nor_A[1]], [-nor_A[1], nor_A[0]]])
        c[i] = np.dot(R[i], nor_B)
        theta[i] = np.arctan2(c[i, 1], c[i, 0])
    return theta

# how to use function 
# input line 1 and line 2 -> outcome = relative angle
Angle_rad = calculate_theta(line1, line2)


# if you have gimbal lock problem -> use this function 
def unwrap_deg(data):
    """
    input = data
    outcome = data without gimbal lock problem
    """
    # Calculate the difference between consecutive data points (angle changes)
    dp = np.diff(data)
    # Adjust the differences to be within the range of -π to π
    # First, add π to dp, then take the modulus with 2π, and subtract π to bring the angle change within the range of -π to π
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    # Handle special case: when the difference is -π, and the original change was positive, fix it to π
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    # Calculate the correction needed (difference between the adjusted angle change and the original angle change)
    dp_corr = dps - dp
    # For angle changes that are smaller than π, we set the correction to 0 (no need to fix)
    dp_corr[np.abs(dp) < np.pi] = 0
    # Accumulate the corrections into the original data starting from the second data point
    data[1:] += np.cumsum(dp_corr)
    # Return the corrected data
    return data

# how to use function 
# input data (angle) -> outcome = fixed angle
fixed_angle = unwrap_deg(Angle_rad)


# change rad to degree
Rankle_sagital_angle = fixed_angle*(180/np.pi)

