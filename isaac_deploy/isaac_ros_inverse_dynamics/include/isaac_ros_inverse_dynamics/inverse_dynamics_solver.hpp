// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "Eigen/Core"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace isaac_ros_inverse_dynamics
{

/// Pinocchio-backed inverse dynamics for an ordered set of controller joints.
///
/// computeInverseDynamics() computes tau = M(q)*a + C(q,v)*v + G(q) via RNEA;
/// computeGravityCompensation() computes the gravity-only term tau = G(q).
///
/// The model is fixed-base (URDF root held fixed, e.g. the pelvis) with the standard
/// downward gravity. Assumes single-DOF joints (idx_qs == idx_vs), which holds for an
/// all-revolute fixed-base model. Both compute methods are allocation-free (pre-sized
/// buffers) and therefore real-time safe, but not safe to call concurrently.
class InverseDynamicsSolver
{
public:
  /// Build the fixed-base model from urdf_path and map joint_names (controller order)
  /// onto the model. Joints absent from the model are skipped (they contribute and
  /// receive zero). Throws std::runtime_error / pinocchio exceptions on a bad URDF,
  /// std::runtime_error if none of the joints are present in the model, and
  /// std::runtime_error if the model has multi-DOF joints (nq != nv).
  InverseDynamicsSolver(
    const std::string & urdf_path,
    const std::vector<std::string> & joint_names);

  /// Number of controller joints; the expected size of q, v, a and tau.
  size_t num_joints() const {return num_joints_;}

  /// Full inverse dynamics (RNEA): tau = M(q)*a + C(q,v)*v + G(q). All four are in
  /// joint_names order and sized num_joints(). Joints absent from the model receive
  /// tau = 0.
  void computeInverseDynamics(
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const Eigen::VectorXd & a,
    Eigen::Ref<Eigen::VectorXd> tau) const;

  /// Gravity compensation only: tau = G(q) (the v = 0, a = 0 special case), computed
  /// via pinocchio::computeGeneralizedGravity. q and tau are in joint_names order and
  /// sized num_joints(); joints absent from the model receive tau = 0.
  void computeGravityCompensation(
    const Eigen::VectorXd & q,
    Eigen::Ref<Eigen::VectorXd> tau) const;

private:
  // ctrl_idx = index within joint_names; pin_idx = Pinocchio idx_vs for that joint.
  struct JointMapping
  {
    size_t ctrl_idx;
    int pin_idx;
  };

  pinocchio::Model model_;
  mutable pinocchio::Data data_;
  std::vector<JointMapping> mappings_;
  size_t num_joints_;
  // Pre-allocated full-model configuration/velocity/acceleration buffers (scratch).
  mutable Eigen::VectorXd q_full_;
  mutable Eigen::VectorXd v_full_;
  mutable Eigen::VectorXd a_full_;
};

}  // namespace isaac_ros_inverse_dynamics
