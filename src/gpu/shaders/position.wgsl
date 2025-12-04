// X and Y Position Force Compute Shaders
// Pulls nodes toward target X or Y positions

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

// Per-node target positions and strengths
struct PositionTarget {
  targetX: f32,
  targetY: f32,
  strengthX: f32,
  strengthY: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> targets: array<PositionTarget>;

@compute @workgroup_size(256)
fn applyPositionForces(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  let node = nodes[i];
  let target = targets[i];
  let alpha = params.alpha;

  var vx = node.vx;
  var vy = node.vy;

  // Apply X force if strength is non-zero
  if (target.strengthX != 0.0) {
    let dx = target.targetX - node.x;
    vx = vx + dx * target.strengthX * alpha;
  }

  // Apply Y force if strength is non-zero
  if (target.strengthY != 0.0) {
    let dy = target.targetY - node.y;
    vy = vy + dy * target.strengthY * alpha;
  }

  nodes[i].vx = vx;
  nodes[i].vy = vy;
}

// Radial force: pulls nodes toward a radius from a center point
struct RadialParams {
  centerX: f32,
  centerY: f32,
  radius: f32,
  strength: f32,
}

@group(0) @binding(3) var<storage, read> radialTargets: array<RadialParams>;

@compute @workgroup_size(256)
fn applyRadialForce(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  let node = nodes[i];
  let radial = radialTargets[i];
  let alpha = params.alpha;

  let dx = node.x - radial.centerX;
  let dy = node.y - radial.centerY;
  let r = sqrt(dx * dx + dy * dy);

  if (r == 0.0) {
    return;
  }

  // Force proportional to (targetRadius - currentRadius)
  let k = (radial.radius - r) * radial.strength * alpha / r;

  nodes[i].vx = node.vx + dx * k;
  nodes[i].vy = node.vy + dy * k;
}
