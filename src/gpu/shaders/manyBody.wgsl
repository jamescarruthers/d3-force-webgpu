// Many-Body Force Compute Shader
// Implements N-body gravitational/repulsive force using tile-based algorithm
// This is O(n²) but highly parallelized, faster than Barnes-Hut for n < ~50k nodes

struct Node {
  x: f32,
  y: f32,
  vx: f32,
  vy: f32,
  fx: f32,      // fixed x (NaN if not fixed)
  fy: f32,      // fixed y (NaN if not fixed)
  strength: f32, // individual node strength (negative = repulsion)
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

const TILE_SIZE: u32 = 256u;

var<workgroup> tile: array<vec4<f32>, 256>; // x, y, strength, padding

// Simple LCG for jiggle (deterministic based on thread id)
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

  if (i >= nodeCount) {
    return;
  }

  let node = nodes[i];
  var forceX: f32 = 0.0;
  var forceY: f32 = 0.0;

  let myPos = vec2<f32>(node.x, node.y);
  let alpha = params.alpha;
  let distanceMin2 = params.distanceMin2;
  let distanceMax2 = params.distanceMax2;

  // Process all other nodes in tiles for better cache efficiency
  let numTiles = (nodeCount + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t++) {
    // Load tile into shared memory
    let tileIdx = t * TILE_SIZE + local_id.x;
    if (tileIdx < nodeCount) {
      let other = nodes[tileIdx];
      tile[local_id.x] = vec4<f32>(other.x, other.y, other.strength, 0.0);
    } else {
      tile[local_id.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    workgroupBarrier();

    // Compute forces from this tile
    let tileEnd = min(TILE_SIZE, nodeCount - t * TILE_SIZE);
    for (var j: u32 = 0u; j < tileEnd; j++) {
      let otherIdx = t * TILE_SIZE + j;
      if (otherIdx == i) {
        continue; // Skip self
      }

      let other = tile[j];
      var dx = other.x - myPos.x;
      var dy = other.y - myPos.y;
      var l2 = dx * dx + dy * dy;

      // Skip if too far
      if (l2 >= distanceMax2) {
        continue;
      }

      // Jiggle coincident nodes
      if (dx == 0.0) {
        dx = jiggle(i * nodeCount + otherIdx);
        l2 += dx * dx;
      }
      if (dy == 0.0) {
        dy = jiggle(i * nodeCount + otherIdx + 1u);
        l2 += dy * dy;
      }

      // Clamp minimum distance to avoid singularities
      if (l2 < distanceMin2) {
        l2 = sqrt(distanceMin2 * l2);
      }

      // Apply force: F = strength * alpha / distance
      // Negative strength = repulsion (push apart)
      let strength = other.z; // other node's strength
      let force = strength * alpha / l2;

      forceX += dx * force;
      forceY += dy * force;
    }

    workgroupBarrier();
  }

  // Update velocities (not positions - that happens in integration step)
  nodes[i].vx = node.vx + forceX;
  nodes[i].vy = node.vy + forceY;
}
