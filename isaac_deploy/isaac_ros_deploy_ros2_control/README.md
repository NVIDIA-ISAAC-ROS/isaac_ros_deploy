# Isaac ROS Deploy ros2_control

## InferenceController Debug Topics

`InferenceController` can optionally publish flattened model inputs and a selected model output while it continues to run through ros2_control command interfaces. These topics are disabled by default.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `publish_debug_topics` | `bool` | `false` | Enables `~/debug_observation` and `~/debug_action`. |
| `debug_action_output_name` | `string` | `arm_action` | Name of the model output tensor to publish on `~/debug_action`. |
| `log_debug_to_console` | `bool` | `false` | Also logs the flattened observation and selected action values to the controller logger. |

### Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/debug_observation` | `std_msgs/msg/Float64MultiArray` | Flattened non-feedback model inputs in model input order. |
| `~/debug_action` | `std_msgs/msg/Float64MultiArray` | Flattened selected model output tensor named by `debug_action_output_name`. |

## SafetyController Command Introspection

`SafetyController` can optionally publish the blend-ratio-scaled joint-position delta that it computes from the upstream command and current measured position. This topic is disabled by default.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `publish_scaled_joint_delta` | `bool` | `false` | Enables publishing `~/scaled_joint_delta`. |
| `scaled_joint_delta_topic` | `string` | `~/scaled_joint_delta` | Topic used for the scaled joint delta output. |

### Topic

| Topic | Type | Description |
| --- | --- | --- |
| `~/scaled_joint_delta` | `isaac_ros_deploy_interfaces/msg/JointCommand` | `position` contains `blend_ratio * (command_position - current_position)` for each configured joint. Other fields are published as `NaN`. |
