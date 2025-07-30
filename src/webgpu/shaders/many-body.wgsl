struct Node {
  position: vec2<f32>,
  velocity: vec2<f32>,
  fixedPosition: vec2<f32>,
  index: f32,
  _padding: f32,
}

struct ForceParams {
  strength: f32,
  distanceMin2: f32,
  distanceMax2: f32,
  theta2: f32,
  alpha: f32,
  nodeCount: f32,
  _padding: vec2<f32>,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> strengths: array<f32>;
@group(0) @binding(2) var<uniform> params: ForceParams;

const EPSILON: f32 = 1e-6;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let nodeCount = u32(params.nodeCount);
  
  if (idx >= nodeCount) {
    return;
  }
  
  var node = nodes[idx];
  var force = vec2<f32>(0.0, 0.0);
  
  // Apply forces from all other nodes
  for (var j = 0u; j < nodeCount; j++) {
    if (j == idx) {
      continue;
    }
    
    let other = nodes[j];
    var delta = node.position - other.position;
    var l = dot(delta, delta);
    
    // Add jiggle if nodes are coincident
    if (l < EPSILON) {
      delta = vec2<f32>(
        (f32(idx) * 0.618033988749895 - floor(f32(idx) * 0.618033988749895)) * 2.0 - 1.0,
        (f32(j) * 0.618033988749895 - floor(f32(j) * 0.618033988749895)) * 2.0 - 1.0
      ) * EPSILON;
      l = dot(delta, delta);
    }
    
    // Skip if beyond max distance
    if (l >= params.distanceMax2) {
      continue;
    }
    
    // Enforce minimum distance
    if (l < params.distanceMin2) {
      l = sqrt(params.distanceMin2 * l);
    } else {
      l = sqrt(l);
    }
    
    // Calculate force magnitude
    let strength = strengths[j] * params.alpha / l;
    force += delta / l * strength;
  }
  
  // Apply force to velocity
  node.velocity += force;
  nodes[idx] = node;
}