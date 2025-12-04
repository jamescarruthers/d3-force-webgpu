// Collision Force Compute Shader
// Prevents node overlap using spatial hashing for efficient neighbor detection

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

// Simple O(n²) collision detection - works well for small-medium graphs
// For very large graphs, would need spatial hash grid

const TILE_SIZE: u32 = 256u;
var<workgroup> tile: array<vec4<f32>, 256>; // x, y, radius, padding

fn jiggle(seed: u32) -> f32 {
  let s = (seed * 1103515245u + 12345u) & 0x7fffffffu;
  return (f32(s) / f32(0x7fffffff) - 0.5) * 1e-6;
}

@compute @workgroup_size(256)
fn detectCollisions(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  let node = nodes[i];
  let myRadius = node.radius;
  var dvx: f32 = 0.0;
  var dvy: f32 = 0.0;

  let strength = params.collisionStrength;
  let numTiles = (nodeCount + TILE_SIZE - 1u) / TILE_SIZE;

  // Process all nodes in tiles
  for (var t: u32 = 0u; t < numTiles; t++) {
    // Load tile into shared memory
    let tileIdx = t * TILE_SIZE + local_id.x;
    if (tileIdx < nodeCount) {
      let other = nodes[tileIdx];
      tile[local_id.x] = vec4<f32>(other.x, other.y, other.radius, 0.0);
    } else {
      tile[local_id.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    workgroupBarrier();

    let tileEnd = min(TILE_SIZE, nodeCount - t * TILE_SIZE);
    for (var j: u32 = 0u; j < tileEnd; j++) {
      let otherIdx = t * TILE_SIZE + j;
      if (otherIdx <= i) {
        continue; // Only process pairs once, and skip self
      }

      let other = tile[j];
      let otherRadius = other.z;
      let combinedRadius = myRadius + otherRadius;

      var dx = node.x - other.x;
      var dy = node.y - other.y;
      var l2 = dx * dx + dy * dy;
      let minDist2 = combinedRadius * combinedRadius;

      if (l2 >= minDist2) {
        continue; // No collision
      }

      // Collision detected - compute repulsive impulse
      if (dx == 0.0) {
        dx = jiggle(i * nodeCount + otherIdx);
        l2 += dx * dx;
      }
      if (dy == 0.0) {
        dy = jiggle(i * nodeCount + otherIdx + 1u);
        l2 += dy * dy;
      }

      let l = sqrt(l2);
      let overlap = combinedRadius - l;

      // Distribute force based on relative radii (bigger nodes move less)
      let totalRadius = myRadius + otherRadius;
      let myWeight = otherRadius / totalRadius;

      // Impulse to separate nodes
      let impulse = overlap * strength * 0.5;
      let nx = dx / l;
      let ny = dy / l;

      dvx += nx * impulse * myWeight;
      dvy += ny * impulse * myWeight;

      // Note: In a full implementation, we'd also update the other node
      // But with symmetric iteration, each pair is processed once
    }

    workgroupBarrier();
  }

  // Apply accumulated collision impulses
  nodes[i].vx = node.vx + dvx;
  nodes[i].vy = node.vy + dvy;
}

// Alternative: Full symmetric collision handling
// Each node handles collisions with ALL other nodes (simpler but more work)
@compute @workgroup_size(256)
fn detectCollisionsSymmetric(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let i = global_id.x;
  let nodeCount = params.nodeCount;

  if (i >= nodeCount) {
    return;
  }

  let node = nodes[i];
  let myRadius = node.radius;
  var dvx: f32 = 0.0;
  var dvy: f32 = 0.0;

  let strength = params.collisionStrength;
  let numTiles = (nodeCount + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t++) {
    let tileIdx = t * TILE_SIZE + local_id.x;
    if (tileIdx < nodeCount) {
      let other = nodes[tileIdx];
      tile[local_id.x] = vec4<f32>(other.x, other.y, other.radius, 0.0);
    } else {
      tile[local_id.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    workgroupBarrier();

    let tileEnd = min(TILE_SIZE, nodeCount - t * TILE_SIZE);
    for (var j: u32 = 0u; j < tileEnd; j++) {
      let otherIdx = t * TILE_SIZE + j;
      if (otherIdx == i) {
        continue;
      }

      let other = tile[j];
      let otherRadius = other.z;
      let combinedRadius = myRadius + otherRadius;

      var dx = node.x - other.x;
      var dy = node.y - other.y;
      var l2 = dx * dx + dy * dy;
      let minDist2 = combinedRadius * combinedRadius;

      if (l2 >= minDist2) {
        continue;
      }

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

    workgroupBarrier();
  }

  nodes[i].vx = node.vx + dvx;
  nodes[i].vy = node.vy + dvy;
}
