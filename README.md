# Isaac ROS Deploy

ROS 2 packages for deploying neural network policies trained in NVIDIA Isaac Lab on real and simulated robots.

## Overview

Isaac ROS Deploy provides ROS 2 packages for deploying
neural network policies trained in NVIDIA Isaac Lab on real and simulated robots.

It loads exported ONNX models, runs inference at control-loop rates, and applies safety
constraints to the outputs. The deployment pipeline has three stages:

1. **Export**: Uses LEAPP to export policies from a Python training setup. LEAPP generates
   a graph of neural network models (e.g., ONNX) together with a YAML metadata file
   describing the topology and expected format of model inputs/outputs.
2. **Runtime**: Isaac ROS Deploy loads the exported LEAPP graph and metadata and runs
   inference. Using the metadata it can automatically connect the inference graph to the
   right ROS topics and hardware interfaces.
3. **Safety**: A safety controller wraps the runtime output to ensure safe operation. It
   can blend policies for gradual enabling or detect and abort unsafe movements.

Two runtimes are available:

`ros2_control`:
: Runs inside the ros2_control update loop, reading joint positions, velocities, and IMU
  data directly from hardware state interfaces. Preferred for proprioceptive-only policies
  needing strict real-time deterministic timing.

`ros2_nodes`:
: Uses Triton Inference Server ensemble to run all models from the LEAPP graph. All
  inputs/outputs flow via ROS topics. Supports multi-model LEAPP graphs and feedback
  self-loops.

The repository currently contains the following packages:

`isaac_deploy_core`:
: Standalone C++ library with no ROS dependencies. Provides the core inference controller,
  input/output builders, safety controller, and inference runner abstractions. Real-time
  safe after activation (no allocations in hot path).

`isaac_ros_deploy_ros2_control`:
: ROS 2 controllers that wrap the core library for use with ros2_control. Includes
  `InferenceController`, `SafetyController`, `FreezeController`,
  `DisableController`, and `ImpedanceController`.

`isaac_ros_deploy_converters`:
: Composable ROS 2 nodes (`InputBuilderNode`, `OutputBuilderNode`) and message
  converters for the ros2_nodes runtime.

`isaac_ros_deploy_bringup`:
: Launch files for the ros2_nodes inference pipeline.

> [!Warning]
> Before using Isaac ROS Deploy to control a robot, please read and familiarize yourself
> with the safety information provided by your robot manufacturer.

> Best practices:

> 1. Familiarize yourself with the location of the emergency stop buttons, and be prepared
>    to apply if necessary.
> 2. Before operation, ensure the working area is free of any persons or other potential
>    hazards.
> 3. Always start with `blend_ratio` at `0.0` and increase gradually.
> 4. Test new policies in simulation (MuJoCo) before deploying on real hardware.
> 5. Take extra caution when testing or deploying new features or code.

---

## Documentation

Please visit the Isaac ROS Documentation to learn how to use this repository.

---

## Packages

## Latest

Update 2026-04-30: Initial release
