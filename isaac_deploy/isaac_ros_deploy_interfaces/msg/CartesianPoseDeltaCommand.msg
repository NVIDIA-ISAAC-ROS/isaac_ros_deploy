# Relative Cartesian pose-delta command anchored to the policy observation.
#
# header.stamp identifies the policy inference tick.
# header.frame_id is the reference frame for observation_pose and delta_position.

std_msgs/Header header

# Raw relative translation from the policy, in header.frame_id.
geometry_msgs/Vector3 delta_position

# Raw relative rotation from the policy, represented as an axis-angle vector.
# The decoder converts this vector to a quaternion and left-multiplies it with
# observation_pose.orientation.
geometry_msgs/Vector3 delta_axis_angle

# Exact Cartesian pose observation used to produce this policy action.
geometry_msgs/Pose observation_pose
