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

#include "isaac_ros_inverse_dynamics/inverse_dynamics_solver.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace isaac_ros_inverse_dynamics
{

InverseDynamicsSolver::InverseDynamicsSolver(
  const std::string & urdf_path,
  const std::vector<std::string> & joint_names)
: num_joints_(joint_names.size())
{
  // Throws on a missing/invalid URDF.
  pinocchio::urdf::buildModel(urdf_path, model_);

  // Single-DOF joints only: q is scattered by idx_vs, which equals idx_qs only when nq == nv.
  if (model_.nq != model_.nv) {
    throw std::runtime_error(
            "InverseDynamicsSolver: model from '" + urdf_path + "' has nq (" +
            std::to_string(model_.nq) + ") != nv (" + std::to_string(model_.nv) +
            "); only single-DOF joints are supported.");
  }

  data_ = pinocchio::Data(model_);
  q_full_ = Eigen::VectorXd::Zero(model_.nq);
  v_full_ = Eigen::VectorXd::Zero(model_.nv);
  a_full_ = Eigen::VectorXd::Zero(model_.nv);

  mappings_.reserve(joint_names.size());
  for (size_t i = 0; i < joint_names.size(); ++i) {
    if (!model_.existJointName(joint_names[i])) {
      continue;  // Joint not modeled: contributes and receives zero.
    }
    const auto joint_id = model_.getJointId(joint_names[i]);
    mappings_.push_back({i, static_cast<int>(model_.idx_vs[joint_id])});
  }

  if (mappings_.empty()) {
    throw std::runtime_error(
            "InverseDynamicsSolver: none of the controller joints are present in the "
            "model built from '" + urdf_path + "'");
  }
}

void InverseDynamicsSolver::computeInverseDynamics(
  const Eigen::VectorXd & q,
  const Eigen::VectorXd & v,
  const Eigen::VectorXd & a,
  Eigen::Ref<Eigen::VectorXd> tau) const
{
  // Scatter the controller-ordered inputs into the full-model configuration.
  for (const auto & m : mappings_) {
    q_full_(m.pin_idx) = q(m.ctrl_idx);
    v_full_(m.pin_idx) = v(m.ctrl_idx);
    a_full_(m.pin_idx) = a(m.ctrl_idx);
  }

  pinocchio::rnea(model_, data_, q_full_, v_full_, a_full_);

  // Gather the per-joint torques back into controller order (zero where unmodeled).
  tau.setZero();
  for (const auto & m : mappings_) {
    tau(m.ctrl_idx) = data_.tau(m.pin_idx);
  }
}

void InverseDynamicsSolver::computeGravityCompensation(
  const Eigen::VectorXd & q,
  Eigen::Ref<Eigen::VectorXd> tau) const
{
  // Scatter the controller-ordered positions into the full-model configuration.
  for (const auto & m : mappings_) {
    q_full_(m.pin_idx) = q(m.ctrl_idx);
  }

  // tau = G(q); cheaper and clearer than rnea(q, 0, 0). Result lands in data_.g.
  pinocchio::computeGeneralizedGravity(model_, data_, q_full_);

  // Gather the per-joint gravity torques back into controller order (zero where unmodeled).
  tau.setZero();
  for (const auto & m : mappings_) {
    tau(m.ctrl_idx) = data_.g(m.pin_idx);
  }
}

}  // namespace isaac_ros_inverse_dynamics
