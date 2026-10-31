// ---- args.wgsl: cell count -> indirect dispatch args (own bind group) ----
// Kept apart so the args buffer is never bound as writable storage in the
// dispatches that read it as indirect args.

@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;

fn count() {
  let n = min(atomicLoad(&counters[0]), MAX_CELLS);
  atomicStore(&counters[0], n);
  atomicStore(&counters[4], n);
  args[0] = (n + 255u) / 256u;
  args[1] = 1u;
  args[2] = 1u;
  // instanced tube draw: 4 vertices per cell
  args[4] = 4u;
  args[5] = n;
  args[6] = 0u;
  args[7] = 0u;
}

// after each step: the step counted too
@compute @workgroup_size(1)
fn main() {
  count();
  atomicAdd(&counters[5], 1u);
}

// after colonies are set down (no step taken)
@compute @workgroup_size(1)
fn recount() {
  count();
}
