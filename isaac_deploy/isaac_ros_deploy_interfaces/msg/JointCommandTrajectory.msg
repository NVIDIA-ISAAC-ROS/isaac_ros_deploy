# Chunked joint-command trajectory.  A block of `horizon` per-step commands
# for `names.size()` joints, with uniform time spacing `step_dt`.  Consumers
# are expected to interpolate linearly between steps.  Step i corresponds
# to sim time `header.stamp + i * step_dt`.

std_msgs/Header header

# Uniform spacing between consecutive steps.  Typically the policy's
# scene.dt at inference time, or the dataset's median frame dt for replay.
builtin_interfaces/Duration step_dt

# Joint names — length N.  Defines the column order of every per-step array
# below.
string[] names

# Per-step command fields.  Each field is a row-major-flattened [H, N] array
# where H == `horizon` and N == len(names).  A field MAY be empty to signal
# "not provided"; consumers treat that identically to an empty field on a
# single-step JointCommand.  Non-empty fields MUST all agree on H.
float64[] position
float64[] velocity
float64[] effort
float64[] kp
float64[] kd

# Horizon (number of steps).  Redundant with position.size()/names.size()
# but explicit to catch malformed messages early and to allow all per-step
# fields to be empty (e.g. a kp-only message would otherwise be ambiguous).
uint32 horizon
