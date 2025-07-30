export class ShaderLoader {
  constructor(device) {
    this.device = device;
    this.shaderModules = new Map();
    this.pipelines = new Map();
    this.bindGroupLayouts = new Map();
  }

  createSimulationTickPipeline() {
    const shaderCode = `
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
    node.position.x += node.velocity.x;
    node.velocity.x *= params.velocityDecay;
  } else {
    node.position.x = node.fixedPosition.x;
    node.velocity.x = 0.0;
  }
  
  if (!isFixedY) {
    node.position.y += node.velocity.y;
    node.velocity.y *= params.velocityDecay;
  } else {
    node.position.y = node.fixedPosition.y;
    node.velocity.y = 0.0;
  }
  
  nodes[idx] = node;
}`;

    const shaderModule = this.device.createShaderModule({
      label: 'Simulation Tick Shader',
      code: shaderCode
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      label: 'Simulation Tick Bind Group Layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage'
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform'
          }
        }
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = this.device.createComputePipeline({
      label: 'Simulation Tick Pipeline',
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });

    this.pipelines.set('simulationTick', pipeline);
    this.bindGroupLayouts.set('simulationTick', bindGroupLayout);

    return { pipeline, bindGroupLayout };
  }

  createBindGroup(name, layout, buffers) {
    const entries = buffers.map((buffer, index) => ({
      binding: index,
      resource: { buffer }
    }));

    return this.device.createBindGroup({
      label: `${name} Bind Group`,
      layout,
      entries
    });
  }

  getPipeline(name) {
    return this.pipelines.get(name);
  }

  getBindGroupLayout(name) {
    return this.bindGroupLayouts.get(name);
  }

  destroy() {
    this.shaderModules.clear();
    this.pipelines.clear();
    this.bindGroupLayouts.clear();
  }
}