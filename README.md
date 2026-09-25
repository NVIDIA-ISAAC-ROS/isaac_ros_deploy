# Isaac ROS Deploy

ROS 2 packages for deploying LEAPP-exported neural-network policies on real and simulated robots.

## Overview

Isaac ROS Deploy provides ROS 2 packages for
deploying neural-network policies on real and simulated robots. It can support
a variety of robot-control policies, including policies derived from
[reinforcement learning in Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
and vision-language-action (VLA) policies exported through
[nvidia-isaac/gr00t-leapp-export](https://github.com/nvidia-isaac/gr00t-leapp-export),
as long as the policy can be exported as a
[LEAPP](https://nvidia-isaac.github.io/leapp/) bundle.

Isaac ROS Deploy bridges the gap between a Python training stack and a robot
control system: it loads LEAPP bundles, runs ONNX inference through NVIDIA
Triton, maps policy terms onto ROS topics or `ros2_control` interfaces, and
optionally gates outputs through a safety controller.

Use Isaac ROS Deploy when you need to deploy a policy as either a ROS 2 node
graph or inside a `ros2_control` loop.

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html) to learn how to use this repository.

---

## Latest

Update 2026-08-18: Added Isaac Sim 6.0 deployment support for Unitree G1 AGILE locomotion
policies

## Synchronized simulation observations

`InputBuilderNode` supports an optional `synchronize_observations` parameter.
Enable it with `use_sim_time` when a simulator supplies timestamp-matched policy inputs.
The `inference_graph.launch.py` launch file forwards both arguments.
The graph's `state/` input kinds select synchronized sources.
Their converters must consume `sensor_msgs/Image`, `JointState`, or `Imu` messages.
Noise and feedback retain their existing producer and initialization behavior.
The core input builder continues to own observation history.

For each increasing simulation timestamp, call `~/prepare_observation` before
publishing observation messages or advancing `/clock`.
The `PrepareObservation` response reports whether the existing inference timer
requires that observation. A failed response must stop the exchange.
When images are required, publish every selected state source with that exact
header timestamp. The node retains one snapshot and advances the input builder
only after the timer fires and all selected inputs arrive.
Wait for `~/observation_consumed` with the same timestamp before advancing again.
Use reliable, transient-local QoS with depth one so discovery cannot lose the acknowledgement.
This acknowledgement means the input tensor snapshot was published, not that
model inference or command execution finished.

Synchronized mode waits for the first successful prepare request before consuming
inputs. Timer callbacks during service discovery do not run inference.
Requests before the next timer deadline do not require images.
Restart the node for each episode before resetting simulation time; decreasing
or repeated request timestamps are rejected. Never advance the clock independently
while a selected observation is outstanding.
With synchronization disabled, the node retains its latest-input behavior.
Both modes schedule inference using the node clock.
