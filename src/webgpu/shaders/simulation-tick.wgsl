struct Node {
  position: vec2<f32>,
  velocity: vec2<f32>,
  fixedPosition: vec2<f32>,
  index: f32,
  _padding: f32,
}

struct SimulationParams {
  alpha: f32,
  alphaDecay: f32,
  alphaTarget: f32,
  velocityDecay: f32,
  nodeCount: f32,
  _padding: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: SimulationParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let nodeCount = u32(params.nodeCount);
  
  if (idx >= nodeCount) {
    return;
  }
  
  var node = nodes[idx];
  
  // Check if position is fixed
  let isFixedX = !isnan(node.fixedPosition.x);
  let isFixedY = !isnan(node.fixedPosition.y);
  
  // Update positions based on velocity
  if (!isFixedX) {
    node.position.x += node.velocity.x * params.velocityDecay;
    node.velocity.x *= params.velocityDecay;
  } else {
    node.position.x = node.fixedPosition.x;
    node.velocity.x = 0.0;
  }
  
  if (!isFixedY) {
    node.position.y += node.velocity.y * params.velocityDecay;
    node.velocity.y *= params.velocityDecay;
  } else {
    node.position.y = node.fixedPosition.y;
    node.velocity.y = 0.0;
  }
  
  nodes[idx] = node;
}