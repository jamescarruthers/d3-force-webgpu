// Shader source code as strings
// These are embedded directly to avoid async loading issues

export const manyBodyShader = `
// Many-Body Force Compute Shader
// Implements N-body gravitational/repulsive force using tile-based algorithm

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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

const TILE_SIZE: u32 = 256u;
var<workgroup> tile: array<vec4<f32>, 256>;

fn jiggle(seed: u32) -> f32 {
  let s = (seed * 1103515245u + 12345u) & 0x7fffffffu;
  return (f32(s) / f32(0x7fffffff) - 0.5) * 1e-6;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;
  let isValid = i < nodeCount;

  // Load node data (use defaults for out-of-bounds threads)
  var myPos = vec2<f32>(0.0, 0.0);
  var myVx: f32 = 0.0;
  var myVy: f32 = 0.0;

  if (isValid) {
    let node = nodes[i];
    myPos = vec2<f32>(node.x, node.y);
    myVx = node.vx;
    myVy = node.vy;
  }

  var forceX: f32 = 0.0;
  var forceY: f32 = 0.0;

  let alpha = params.alpha;
  let distanceMin2 = params.distanceMin2;
  let distanceMax2 = params.distanceMax2;

  let numTiles = (nodeCount + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t++) {
    // All threads participate in loading the tile
    let tileIdx = t * TILE_SIZE + local_id.x;
    if (tileIdx < nodeCount) {
      let other = nodes[tileIdx];
      tile[local_id.x] = vec4<f32>(other.x, other.y, other.strength, 0.0);
    } else {
      tile[local_id.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // All threads must hit this barrier
    workgroupBarrier();

    // Only valid threads compute forces
    if (isValid) {
      let tileEnd = min(TILE_SIZE, nodeCount - t * TILE_SIZE);
      for (var j: u32 = 0u; j < tileEnd; j++) {
        let otherIdx = t * TILE_SIZE + j;
        if (otherIdx != i) {
          let other = tile[j];
          var dx = other.x - myPos.x;
          var dy = other.y - myPos.y;
          var l2 = dx * dx + dy * dy;

          if (l2 < distanceMax2) {
            if (dx == 0.0) {
              dx = jiggle(i * nodeCount + otherIdx);
              l2 += dx * dx;
            }
            if (dy == 0.0) {
              dy = jiggle(i * nodeCount + otherIdx + 1u);
              l2 += dy * dy;
            }

            if (l2 < distanceMin2) {
              l2 = sqrt(distanceMin2 * l2);
            }

            let strength = other.z;
            let force = strength * alpha / l2;

            forceX += dx * force;
            forceY += dy * force;
          }
        }
      }
    }

    // All threads must hit this barrier
    workgroupBarrier();
  }

  // Only valid threads write results
  if (isValid) {
    nodes[i].vx = myVx + forceX;
    nodes[i].vy = myVy + forceY;
  }
}
`;

export const linkShader = `
// Link Force Compute Shader

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
  sourceIdx: f32,
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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> links: array<Link>;
@group(0) @binding(2) var<uniform> params: Params;

fn jiggle(seed: u32) -> f32 {
  let s = (seed * 1103515245u + 12345u) & 0x7fffffffu;
  return (f32(s) / f32(0x7fffffff) - 0.5) * 1e-6;
}

@compute @workgroup_size(256)
fn main(
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

  for (var i: u32 = 0u; i < linkCount; i++) {
    let link = links[i];
    let sourceIdx = u32(link.sourceIdx);
    let targetIdx = u32(link.targetIdx);

    if (sourceIdx != nodeIdx && targetIdx != nodeIdx) {
      continue;
    }

    let srcNode = nodes[sourceIdx];
    let dstNode = nodes[targetIdx];

    var dx = dstNode.x + dstNode.vx - srcNode.x - srcNode.vx;
    var dy = dstNode.y + dstNode.vy - srcNode.y - srcNode.vy;

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
`;

export const integrateShader = `
// Position Integration Compute Shader

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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

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

  if (isNaN(node.fx)) {
    node.vx = node.vx * velocityDecay;
    node.x = node.x + node.vx;
  } else {
    node.x = node.fx;
    node.vx = 0.0;
  }

  if (isNaN(node.fy)) {
    node.vy = node.vy * velocityDecay;
    node.y = node.y + node.vy;
  } else {
    node.y = node.fy;
    node.vy = 0.0;
  }

  nodes[i] = node;
}
`;

export const collisionShader = `
// Collision Force Compute Shader

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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

const TILE_SIZE: u32 = 256u;
var<workgroup> tile: array<vec4<f32>, 256>;

fn jiggle(seed: u32) -> f32 {
  let s = (seed * 1103515245u + 12345u) & 0x7fffffffu;
  return (f32(s) / f32(0x7fffffff) - 0.5) * 1e-6;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;
  let isValid = i < nodeCount;

  // Load node data (use defaults for out-of-bounds threads)
  var myX: f32 = 0.0;
  var myY: f32 = 0.0;
  var myRadius: f32 = 0.0;
  var myVx: f32 = 0.0;
  var myVy: f32 = 0.0;

  if (isValid) {
    let node = nodes[i];
    myX = node.x;
    myY = node.y;
    myRadius = node.radius;
    myVx = node.vx;
    myVy = node.vy;
  }

  var dvx: f32 = 0.0;
  var dvy: f32 = 0.0;

  let strength = params.collisionStrength;
  let numTiles = (nodeCount + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t++) {
    // All threads participate in loading the tile
    let tileIdx = t * TILE_SIZE + local_id.x;
    if (tileIdx < nodeCount) {
      let other = nodes[tileIdx];
      tile[local_id.x] = vec4<f32>(other.x, other.y, other.radius, 0.0);
    } else {
      tile[local_id.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // All threads must hit this barrier
    workgroupBarrier();

    // Only valid threads compute collisions
    if (isValid) {
      let tileEnd = min(TILE_SIZE, nodeCount - t * TILE_SIZE);
      for (var j: u32 = 0u; j < tileEnd; j++) {
        let otherIdx = t * TILE_SIZE + j;
        if (otherIdx != i) {
          let other = tile[j];
          let otherRadius = other.z;
          let combinedRadius = myRadius + otherRadius;

          var dx = myX - other.x;
          var dy = myY - other.y;
          var l2 = dx * dx + dy * dy;
          let minDist2 = combinedRadius * combinedRadius;

          if (l2 < minDist2) {
            if (dx == 0.0) {
              dx = jiggle(min(i, otherIdx) * nodeCount + max(i, otherIdx));
              l2 += dx * dx;
            }
            if (dy == 0.0) {
              dy = jiggle(min(i, otherIdx) * nodeCount + max(i, otherIdx) + 1u);
              l2 += dy * dy;
            }

            let l = sqrt(l2);
            let overlap = combinedRadius - l;

            let totalRadius = myRadius + otherRadius;
            let myWeight = otherRadius / totalRadius;

            let impulse = overlap * strength * 0.5;
            let nx = dx / l;
            let ny = dy / l;

            dvx += nx * impulse * myWeight;
            dvy += ny * impulse * myWeight;
          }
        }
      }
    }

    // All threads must hit this barrier
    workgroupBarrier();
  }

  // Only valid threads write results
  if (isValid) {
    nodes[i].vx = myVx + dvx;
    nodes[i].vy = myVy + dvy;
  }
}
`;

export const forceXShader = `
// X-Positioning Force Compute Shader
// Pushes nodes toward a target x position

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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  if (i >= params.nodeCount) {
    return;
  }

  let node = nodes[i];
  let alpha = params.alpha;
  let targetX = params.forceXTarget;
  let strength = params.forceXStrength;

  let dx = targetX - node.x;
  nodes[i].vx = node.vx + dx * strength * alpha;
}
`;

export const forceYShader = `
// Y-Positioning Force Compute Shader
// Pushes nodes toward a target y position

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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  if (i >= params.nodeCount) {
    return;
  }

  let node = nodes[i];
  let alpha = params.alpha;
  let targetY = params.forceYTarget;
  let strength = params.forceYStrength;

  let dy = targetY - node.y;
  nodes[i].vy = node.vy + dy * strength * alpha;
}
`;

export const forceRadialShader = `
// Radial Force Compute Shader
// Pushes nodes toward a target radius from a center point

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
  forceXTarget: f32,
  forceXStrength: f32,
  forceYTarget: f32,
  forceYStrength: f32,
  radialX: f32,
  radialY: f32,
  radialRadius: f32,
  radialStrength: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  if (i >= params.nodeCount) {
    return;
  }

  let node = nodes[i];
  let alpha = params.alpha;
  let cx = params.radialX;
  let cy = params.radialY;
  let targetRadius = params.radialRadius;
  let strength = params.radialStrength;

  let dx = node.x - cx;
  let dy = node.y - cy;
  let r = sqrt(dx * dx + dy * dy);

  if (r > 0.0) {
    let k = (targetRadius - r) * strength * alpha / r;
    nodes[i].vx = node.vx + dx * k;
    nodes[i].vy = node.vy + dy * k;
  }
}
`;
