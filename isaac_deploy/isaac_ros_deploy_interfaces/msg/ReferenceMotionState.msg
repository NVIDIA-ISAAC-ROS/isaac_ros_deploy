# Full reference motion state at current time.

std_msgs/Header header

# Elapsed time since start/reset (seconds).
float64 elapsed_time

# Motion duration (seconds).
float64 motion_duration

# Current motion progress (0.0 to 1.0, wraps in loop mode).
float64 progress

# Whether motion is looping.
bool is_looping

# Body names in the motion.
string[] body_names

# Joint names in the motion.
string[] joint_names

# Body poses (position + orientation) in reference motion frame.
geometry_msgs/Pose[] body_poses

# Body twists (linear + angular velocity) in reference motion frame.
geometry_msgs/Twist[] body_twists

# Joint positions [num_joints].
float64[] joint_positions

# Joint velocities [num_joints].
float64[] joint_velocities

# Contact flags [num_contacts].
float64[] contacts
