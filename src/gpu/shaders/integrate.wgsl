// Position Integration Compute Shader
// Applies velocity Verlet integration to update node positions
// Handles fixed positions (fx, fy) and velocity decay

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

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

// Check if a value is NaN (used for fixed position detection)
fn isNaN(v: f32) -> bool {
  return !(v == v);
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  var node = nodes[i];
  let velocityDecay = params.velocityDecay;

  // Update x position
  if (isNaN(node.fx)) {
    // Not fixed: apply velocity with decay
    node.vx = node.vx * velocityDecay;
    node.x = node.x + node.vx;
  } else {
    // Fixed: set position and zero velocity
    node.x = node.fx;
    node.vx = 0.0;
  }

  // Update y position
  if (isNaN(node.fy)) {
    node.vy = node.vy * velocityDecay;
    node.y = node.y + node.vy;
  } else {
    node.y = node.fy;
    node.vy = 0.0;
  }

  nodes[i] = node;
}
