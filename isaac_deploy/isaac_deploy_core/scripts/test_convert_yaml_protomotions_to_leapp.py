#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate protomotions-to-leapp YAML conversion output."""

import argparse
from pathlib import Path
import sys

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from convert_yaml_protomotions_to_leapp import convert  # noqa: E402


def _load_yaml(path: Path) -> dict:
    """Load YAML file and return as dict."""
    return yaml.safe_load(path.read_text())


def _normalize(data):
    """Recursively normalize data for comparison."""
    if isinstance(data, dict):
        return {k: _normalize(v) for k, v in sorted(data.items())}
    if isinstance(data, list):
        return [_normalize(v) for v in data]
    return data


def assert_conversion_matches(protomotions_yaml: Path, leapp_yaml: Path) -> None:
    """Convert protomotions YAML and compare to expected leapp YAML."""
    old_path = protomotions_yaml.resolve()
    expected_path = leapp_yaml.resolve()

    old_data = _load_yaml(old_path)
    expected_data = _load_yaml(expected_path)

    converted_data = convert(old_data)

    # Normalize both for comparison (dict key order shouldn't matter).
    assert _normalize(converted_data) == _normalize(expected_data), (
        'Converted output does not match the expected leapp YAML. '
        'If the old format changed, re-run the converter script:\n'
        '  python3 scripts/convert_yaml_protomotions_to_leapp.py '
        f'{old_path} -o {expected_path}'
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify converted protomotions YAML matches a checked-in '
            'LEAPP YAML.'
        )
    )
    parser.add_argument(
        '--protomotions-yaml',
        type=Path,
        required=True,
        help='Path to the protomotions YAML file to convert.',
    )
    parser.add_argument(
        '--leapp-yaml',
        type=Path,
        required=True,
        help='Path to the expected LEAPP YAML file.',
    )
    return parser.parse_args()


def main() -> int:
    """Run the CLI check and return process exit code."""
    args = _parse_args()
    assert_conversion_matches(args.protomotions_yaml, args.leapp_yaml)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
