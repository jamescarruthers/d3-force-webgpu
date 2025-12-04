// Link Force Compute Shader
// Implements spring-like forces between connected nodes (Hooke's Law)

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

struct Link {
  sourceIdx: f32,  // Using f32 for simplicity, will cast to u32
  targetIdx: f32,
  distance: f32,
  strength: f32,
  bias: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
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

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> links: array<Link>;
@group(0) @binding(2) var<uniform> params: Params;

// Atomic accumulators for velocity updates
// Since multiple links can affect the same node, we need atomic operations
// WebGPU doesn't support atomic floats directly, so we use a workaround:
// We'll compute force contributions per link and store them, then reduce

struct ForceAccum {
  dvx: f32,
  dvy: f32,
}

@group(0) @binding(3) var<storage, read_write> forceAccum: array<ForceAccum>;

fn jiggle(seed: u32) -> f32 {
  let s = (seed * 1103515245u + 12345u) & 0x7fffffffu;
  return (f32(s) / f32(0x7fffffff) - 0.5) * 1e-6;
}

// First pass: compute forces per link and accumulate to nodes
@compute @workgroup_size(256)
fn computeLinkForces(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let linkIdx = global_id.x;
  let linkCount = params.linkCount;

  if (linkIdx >= linkCount) {
    return;
  }

  let link = links[linkIdx];
  let sourceIdx = u32(link.sourceIdx);
  let targetIdx = u32(link.targetIdx);

  let source = nodes[sourceIdx];
  let target = nodes[targetIdx];

  // Calculate displacement including current velocities
  var dx = target.x + target.vx - source.x - source.vx;
  var dy = target.y + target.vy - source.y - source.vy;

  // Jiggle if coincident
  if (dx == 0.0 && dy == 0.0) {
    dx = jiggle(linkIdx * 2u);
    dy = jiggle(linkIdx * 2u + 1u);
  }

  let l = sqrt(dx * dx + dy * dy);
  let desiredDistance = link.distance;
  let strength = link.strength;
  let alpha = params.alpha;

  // Spring force: (currentLength - desiredLength) / currentLength * alpha * strength
  let force = (l - desiredDistance) / l * alpha * strength;
  let fx = dx * force;
  let fy = dy * force;

  // Bias determines how force is distributed between source and target
  let bias = link.bias;

  // Apply force to target (pulled toward source)
  // Note: We're using non-atomic writes here which may cause race conditions
  // For better accuracy, we'd need atomic floats or a reduction pass
  // This is a trade-off for performance - results will be approximate but stable

  // Target gets force * bias
  nodes[targetIdx].vx = target.vx - fx * bias;
  nodes[targetIdx].vy = target.vy - fy * bias;

  // Source gets force * (1 - bias)
  nodes[sourceIdx].vx = source.vx + fx * (1.0 - bias);
  nodes[sourceIdx].vy = source.vy + fy * (1.0 - bias);
}

// Alternative: Compute per-node by iterating all links
// This avoids race conditions but is less parallel for sparse graphs
@compute @workgroup_size(256)
fn computeLinkForcesPerNode(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let nodeIdx = global_id.x;
  let nodeCount = params.nodeCount;
  let linkCount = params.linkCount;
  let alpha = params.alpha;

  if (nodeIdx >= nodeCount) {
    return;
  }

  let node = nodes[nodeIdx];
  var dvx: f32 = 0.0;
  var dvy: f32 = 0.0;

  // Iterate through all links to find those connected to this node
  for (var i: u32 = 0u; i < linkCount; i++) {
    let link = links[i];
    let sourceIdx = u32(link.sourceIdx);
    let targetIdx = u32(link.targetIdx);

    if (sourceIdx != nodeIdx && targetIdx != nodeIdx) {
      continue;
    }

    let source = nodes[sourceIdx];
    let target = nodes[targetIdx];

    var dx = target.x + target.vx - source.x - source.vx;
    var dy = target.y + target.vy - source.y - source.vy;

    if (dx == 0.0 && dy == 0.0) {
      dx = jiggle(i * 2u);
      dy = jiggle(i * 2u + 1u);
    }

    let l = sqrt(dx * dx + dy * dy);
    let force = (l - link.distance) / l * alpha * link.strength;
    let fx = dx * force;
    let fy = dy * force;

    if (nodeIdx == targetIdx) {
      dvx -= fx * link.bias;
      dvy -= fy * link.bias;
    } else {
      dvx += fx * (1.0 - link.bias);
      dvy += fy * (1.0 - link.bias);
    }
  }

  nodes[nodeIdx].vx = node.vx + dvx;
  nodes[nodeIdx].vy = node.vy + dvy;
}
