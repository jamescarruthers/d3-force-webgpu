// Center Force Compute Shader
// Centers all nodes around a specified point
// Also includes X and Y positioning forces

struct Node {
  x: f32,
  y: f32,
  vx: f32,
  vy: f32,
  fx: f32,
  fy: f32,
  strength: f32,
  radius: f32,
}

struct Params {
  alpha: f32,
  velocityDecay: f32,
  nodeCount: u32,
  linkCount: u32,
  centerX: f32,
  centerY: f32,
  centerStrength: f32,
  theta2: f32,
  distanceMin2: f32,
  distanceMax2: f32,
  iterations: u32,
  collisionRadius: f32,
  collisionStrength: f32,
  collisionIterations: u32,
  _pad1: f32,
  _pad2: f32,
}

struct CenterParams {
  targetX: f32,
  targetY: f32,
  strength: f32,
  nodeCount: u32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

// Shared memory for parallel reduction
var<workgroup> sharedSumX: array<f32, 256>;
var<workgroup> sharedSumY: array<f32, 256>;
var<workgroup> sharedCount: array<f32, 256>;

// Partial sums buffer for multi-workgroup reduction
@group(0) @binding(2) var<storage, read_write> partialSums: array<f32>;

// First pass: compute partial sums per workgroup
@compute @workgroup_size(256)
fn computeCenterSum(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;
  let lid = local_id.x;

  // Load node position or zero
  if (i < nodeCount) {
    let node = nodes[i];
    sharedSumX[lid] = node.x;
    sharedSumY[lid] = node.y;
    sharedCount[lid] = 1.0;
  } else {
    sharedSumX[lid] = 0.0;
    sharedSumY[lid] = 0.0;
    sharedCount[lid] = 0.0;
  }

  workgroupBarrier();

  // Parallel reduction within workgroup
  for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid < stride) {
      sharedSumX[lid] = sharedSumX[lid] + sharedSumX[lid + stride];
      sharedSumY[lid] = sharedSumY[lid] + sharedSumY[lid + stride];
      sharedCount[lid] = sharedCount[lid] + sharedCount[lid + stride];
    }
    workgroupBarrier();
  }

  // Write workgroup result to global memory
  if (lid == 0u) {
    let wgIdx = workgroup_id.x;
    partialSums[wgIdx * 3u + 0u] = sharedSumX[0];
    partialSums[wgIdx * 3u + 1u] = sharedSumY[0];
    partialSums[wgIdx * 3u + 2u] = sharedCount[0];
  }
}

// Second pass: apply center offset to all nodes
// Assumes partialSums[0], [1], [2] contain total sumX, sumY, count
@compute @workgroup_size(256)
fn applyCenter(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  // Read computed center from partial sums (first 3 elements after final reduction)
  let totalX = partialSums[0];
  let totalY = partialSums[1];
  let count = partialSums[2];

  if (count == 0.0) {
    return;
  }

  let avgX = totalX / count;
  let avgY = totalY / count;

  // Offset to move center to target
  let offsetX = params.centerX - avgX;
  let offsetY = params.centerY - avgY;

  // Apply offset scaled by strength
  let strength = params.centerStrength;

  var node = nodes[i];
  node.x = node.x + offsetX * strength;
  node.y = node.y + offsetY * strength;
  nodes[i] = node;
}

// Alternative: Single-pass center force (simpler but requires known center)
// Use when center offset is precomputed on CPU
struct CenterOffset {
  offsetX: f32,
  offsetY: f32,
  strength: f32,
  _pad: f32,
}

@group(0) @binding(3) var<uniform> centerOffset: CenterOffset;

@compute @workgroup_size(256)
fn applyCenterOffset(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  var node = nodes[i];
  node.x = node.x + centerOffset.offsetX * centerOffset.strength;
  node.y = node.y + centerOffset.offsetY * centerOffset.strength;
  nodes[i] = node;
}
