# Joint command message for reference motion tracking.
# All arrays must have the same size as names[].

std_msgs/Header header

# Joint names (determines array ordering).
string[] names

# Commanded joint positions (radians for revolute, meters for prismatic).
float64[] position

# Commanded joint velocities (rad/s or m/s).
float64[] velocity

# Commanded joint efforts (Nm or N).
float64[] effort

# Position gain (optional, same size as names or empty).
float64[] kp

# Derivative gain (optional, same size as names or empty).
float64[] kd
