#!/usr/bin/env python3

# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for LEAPP YAML to Triton repository conversion."""

from pathlib import Path

from isaac_ros_deploy_converters import create_triton_model_repo as repo
import pytest
import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _single_model_config(backend: str | None = "onnx") -> dict:
    config = {
        "models": {
            "policy": {
                "inputs": [
                    {"name": "external", "dtype": "float32", "shape": [1, 1]},
                    {"name": "feedback", "dtype": "float32", "shape": [1, 1]},
                ],
                "outputs": [
                    {"name": "feedback_out", "dtype": "float32", "shape": [1, 1]},
                    {"name": "action", "dtype": "float32", "shape": [1, 1]},
                ],
                "parameters": {
                    "model_path": "policy.onnx",
                },
            }
        },
        "pipeline": {
            "inputs": {"policy": ["external"]},
            "outputs": {"policy": ["action"]},
            "feedback_flow": {"policy/feedback_out": ["policy/feedback"]},
            "data_flow": {},
        },
    }
    if backend is not None:
        config["models"]["policy"]["parameters"]["backend"] = backend
    return config


def _multi_model_config() -> dict:
    return {
        "models": {
            "encoder": {
                "inputs": [
                    {"name": "state", "dtype": "float32", "shape": [1, 2]},
                ],
                "outputs": [
                    {"name": "latent", "dtype": "float32", "shape": [1, 4]},
                ],
                "parameters": {
                    "backend": "onnx",
                    "model_path": "encoder.onnx",
                },
            },
            "policy": {
                "inputs": [
                    {"name": "latent", "dtype": "float32", "shape": [1, 4]},
                    {"name": "command", "dtype": "float32", "shape": [1, 1]},
                ],
                "outputs": [
                    {"name": "action", "dtype": "float32", "shape": [1, 3]},
                ],
                "parameters": {
                    "backend": "onnx",
                    "model_path": "policy.onnx",
                },
            },
        },
        "pipeline": {
            "inputs": {
                "encoder": ["state"],
                "policy": ["command"],
            },
            "outputs": {"policy": ["action"]},
            "feedback_flow": {},
            "data_flow": {"encoder/latent": ["policy/latent"]},
        },
    }


def _multi_model_collision_config() -> dict:
    return {
        "models": {
            "producer": {
                "inputs": [
                    {"name": "state", "dtype": "float32", "shape": [1, 2]},
                ],
                "outputs": [
                    {"name": "shared", "dtype": "float32", "shape": [1, 4]},
                ],
                "parameters": {
                    "backend": "onnx",
                    "model_path": "producer.onnx",
                },
            },
            "consumer": {
                "inputs": [
                    {"name": "shared", "dtype": "float32", "shape": [1, 4]},
                ],
                "outputs": [
                    {"name": "action", "dtype": "float32", "shape": [1, 3]},
                ],
                "parameters": {
                    "backend": "onnx",
                    "model_path": "consumer.onnx",
                },
            },
        },
        "pipeline": {
            "inputs": {
                "producer": ["state"],
                "consumer": ["shared"],
            },
            "outputs": {
                "producer": ["shared"],
                "consumer": ["action"],
            },
            "feedback_flow": {},
            "data_flow": {},
        },
    }


def test_feedback_flow_contributes_to_ensemble_io(tmp_path, monkeypatch):
    """Feedback source/target tensors should come from upstream feedback_flow."""
    monkeypatch.setattr(repo, "_create_model_dir", lambda *args, **kwargs: False)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _single_model_config())

    result = repo.create_triton_model_repo(config_path, tmp_path / "repo")

    assert result.input_tensor_names == ["external", "feedback"]
    assert result.output_tensor_names == ["feedback_out", "action"]


def test_model_config_pins_model_to_requested_device():
    """Generated configs should not let Triton spread across visible GPUs."""
    config = repo._generate_model_config(
        "policy",
        inputs=[{"name": "state", "dtype": "float32", "shape": [1, 3]}],
        outputs=[{"name": "action", "dtype": "float32", "shape": [1, 2]}],
        dynamic_batch=False,
        use_cpu=True,
    )

    assert "instance_group [{ kind: KIND_CPU }]" in config

    default_config = repo._generate_model_config(
        "policy",
        inputs=[{"name": "state", "dtype": "float32", "shape": [1, 3]}],
        outputs=[{"name": "action", "dtype": "float32", "shape": [1, 2]}],
        dynamic_batch=False,
    )

    assert "instance_group [{ kind: KIND_GPU gpus: [0] }]" in default_config


def test_cpu_models_are_runtime_options(tmp_path, monkeypatch):
    """CPU placement should come from the converter API, not the LEAPP YAML."""
    cpu_by_model = {}

    def _record_create(*args, **kwargs):
        cpu_by_model[args[1]] = kwargs["use_cpu"]
        return False

    monkeypatch.setattr(repo, "_create_model_dir", _record_create)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _single_model_config())

    repo.create_triton_model_repo(
        config_path,
        tmp_path / "repo",
        cpu_models={"policy"},
    )

    assert cpu_by_model == {"policy": True}


def test_unknown_cpu_model_is_rejected(tmp_path, monkeypatch):
    """CPU model names should match LEAPP model names."""
    called = False

    def _unexpected_create(*args, **kwargs):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr(repo, "_create_model_dir", _unexpected_create)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _single_model_config())

    with pytest.raises(ValueError, match=r"CPU model.*missing"):
        repo.create_triton_model_repo(
            config_path,
            tmp_path / "repo",
            cpu_models={"missing"},
        )

    assert not called


def test_jit_backend_is_rejected_before_model_repo_creation(tmp_path, monkeypatch):
    """The Triton converter only supports ONNX LEAPP artifacts."""
    called = False

    def _unexpected_create(*args, **kwargs):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr(repo, "_create_model_dir", _unexpected_create)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _single_model_config(backend="jit"))

    with pytest.raises(ValueError, match=r"backend 'jit'.*not supported"):
        repo.create_triton_model_repo(config_path, tmp_path / "repo")

    assert not called


def test_missing_backend_is_rejected_with_clear_message(tmp_path, monkeypatch):
    """Missing LEAPP backend should not leak Python None into operator errors."""
    called = False

    def _unexpected_create(*args, **kwargs):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr(repo, "_create_model_dir", _unexpected_create)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _single_model_config(backend=None))

    with pytest.raises(ValueError, match=r"parameters\.backend.*missing"):
        repo.create_triton_model_repo(config_path, tmp_path / "repo")

    assert not called


def test_external_onnx_data_is_materialized_inside_model_directory(
    tmp_path, monkeypatch
):
    """ONNX artifacts must be zero-copy files inside the Triton model dir."""
    config = _single_model_config()
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config)
    source_model = tmp_path / "policy.onnx"
    source_data = tmp_path / "policy.onnx.data"
    source_model.write_bytes(b"model")
    source_data.write_bytes(b"external weights")
    monkeypatch.setattr(repo, "_has_dynamic_batch", lambda _: False)

    repo._create_model_dir(
        config_path,
        "policy",
        tmp_path / "repo",
        config,
    )

    version_dir = tmp_path / "repo" / "policy" / "1"
    linked_model = version_dir / "model.onnx"
    linked_data = version_dir / "policy.onnx.data"
    assert linked_model.is_symlink()
    assert linked_data.is_symlink()
    assert linked_model.resolve() == source_model.resolve()
    assert linked_data.resolve() == source_data.resolve()


def test_onnx_artifacts_are_never_copied_when_symlink_fails(
    tmp_path, monkeypatch
):
    """ONNX artifacts must not fall back to an expensive copy."""
    config = _single_model_config()
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config)
    (tmp_path / "policy.onnx").write_bytes(b"model")
    (tmp_path / "policy.onnx.data").write_bytes(b"external weights")
    monkeypatch.setattr(repo, "_has_dynamic_batch", lambda _: False)

    def _cannot_symlink(_self, _target):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", _cannot_symlink)

    with pytest.raises(RuntimeError, match=r"symlink ONNX artifact"):
        repo._create_model_dir(
            config_path,
            "policy",
            tmp_path / "repo",
            config,
        )


def test_external_data_is_found_next_to_symlinked_model_artifact(
    tmp_path, monkeypatch
):
    """A Bazel symlink assembly must retain sibling external-data discovery."""
    config = _single_model_config()
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    config_path = asset_dir / "config.yaml"
    _write_yaml(config_path, config)

    model_source = tmp_path / "model_source" / "policy.onnx"
    data_source = tmp_path / "data_source" / "policy.onnx.data"
    model_source.parent.mkdir()
    data_source.parent.mkdir()
    model_source.write_bytes(b"model")
    data_source.write_bytes(b"external weights")
    (asset_dir / "policy.onnx").symlink_to(model_source)
    (asset_dir / "policy.onnx.data").symlink_to(data_source)
    monkeypatch.setattr(repo, "_has_dynamic_batch", lambda _: False)

    repo._create_model_dir(
        config_path,
        "policy",
        tmp_path / "repo",
        config,
    )

    version_dir = tmp_path / "repo" / "policy" / "1"
    linked_model = version_dir / "model.onnx"
    linked_data = version_dir / "policy.onnx.data"
    assert linked_model.is_symlink()
    assert linked_data.is_symlink()
    assert linked_model.resolve() == model_source.resolve()
    assert linked_data.resolve() == data_source.resolve()


def test_multi_model_data_flow_uses_internal_tensor_names(tmp_path, monkeypatch):
    """Hidden model-to-model edges should use private ensemble tensor names."""
    monkeypatch.setattr(repo, "_create_model_dir", lambda *args, **kwargs: False)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _multi_model_config())

    result = repo.create_triton_model_repo(config_path, tmp_path / "repo")
    config_pbtxt = (tmp_path / "repo" / "ensemble" / "config.pbtxt").read_text()

    assert result.input_tensor_names == ["state", "command"]
    assert result.output_tensor_names == ["action"]
    assert 'key: "latent"\n        value: "_internal_latent"' in config_pbtxt


def test_multi_model_input_output_name_collision_renames_input_binding(
    tmp_path, monkeypatch
):
    """Triton ensemble inputs should be renamed when they collide with outputs."""
    monkeypatch.setattr(repo, "_create_model_dir", lambda *args, **kwargs: False)

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _multi_model_collision_config())

    result = repo.create_triton_model_repo(config_path, tmp_path / "repo")
    config_pbtxt = (tmp_path / "repo" / "ensemble" / "config.pbtxt").read_text()

    assert result.input_tensor_names == ["state", "shared"]
    assert result.input_binding_names == ["state", "_in_shared"]
    assert result.output_tensor_names == ["shared", "action"]
    assert 'key: "shared"\n        value: "_in_shared"' in config_pbtxt
    assert 'key: "shared"\n        value: "shared"' in config_pbtxt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
