import constant from "../../constant.js";

export default function() {
  var nodes,
      device,
      pipeline,
      bindGroup,
      strengthBuffer,
      forceParamsBuffer,
      strength = constant(-30),
      strengths,
      distanceMin2 = 1,
      distanceMax2 = Infinity,
      theta2 = 0.81;

  async function force(alpha) {
    if (!device || !pipeline || !bindGroup) return;

    const strengthValue = typeof strength === 'function' ? strength() : strength;
    
    const forceParams = new Float32Array([
      strengthValue,
      distanceMin2,
      distanceMax2,
      theta2,
      alpha,
      nodes.length,
      0, 0
    ]);

    device.queue.writeBuffer(forceParamsBuffer, 0, forceParams);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(nodes.length / 64));
    passEncoder.end();
    
    device.queue.submit([commandEncoder.finish()]);
    
    // Wait for completion
    await device.queue.onSubmittedWorkDone();
  }

  function initialize() {
    if (!nodes) return;
    var i, n = nodes.length, node;
    strengths = new Float32Array(n);
    for (i = 0; i < n; ++i) {
      node = nodes[i];
      strengths[node.index] = +strength(node, i, nodes);
    }
    
    if (device && strengthBuffer) {
      device.queue.writeBuffer(strengthBuffer, 0, strengths);
    }
  }

  force.initialize = function(_nodes, _random, _device, nodeBuffer) {
    nodes = _nodes;
    device = _device;
    initialize();

    const shaderCode = `
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
  
  for (var j = 0u; j < nodeCount; j++) {
    if (j == idx) {
      continue;
    }
    
    let other = nodes[j];
    var delta = node.position - other.position;
    var l = dot(delta, delta);
    
    if (l < EPSILON) {
      delta = vec2<f32>(
        (f32(idx) * 0.618033988749895 - floor(f32(idx) * 0.618033988749895)) * 2.0 - 1.0,
        (f32(j) * 0.618033988749895 - floor(f32(j) * 0.618033988749895)) * 2.0 - 1.0
      ) * EPSILON;
      l = dot(delta, delta);
    }
    
    if (l >= params.distanceMax2) {
      continue;
    }
    
    if (l < params.distanceMin2) {
      l = sqrt(params.distanceMin2 * l);
    } else {
      l = sqrt(l);
    }
    
    let strength = strengths[j] * params.alpha / l;
    force += delta / l * strength;
  }
  
  node.velocity += force;
  nodes[idx] = node;
}`;

    const shaderModule = device.createShaderModule({
      label: 'Many Body Force Shader',
      code: shaderCode
    });

    strengthBuffer = device.createBuffer({
      label: 'Strength Buffer',
      size: strengths.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(strengthBuffer.getMappedRange()).set(strengths);
    strengthBuffer.unmap();

    forceParamsBuffer = device.createBuffer({
      label: 'Force Parameters',
      size: 8 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' }
        }
      ]
    });

    pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: nodeBuffer } },
        { binding: 1, resource: { buffer: strengthBuffer } },
        { binding: 2, resource: { buffer: forceParamsBuffer } }
      ]
    });
  };

  force.strength = function(_) {
    return arguments.length ? (strength = typeof _ === "function" ? _ : constant(+_), initialize(), force) : strength;
  };

  force.distanceMin = function(_) {
    return arguments.length ? (distanceMin2 = _ * _, force) : Math.sqrt(distanceMin2);
  };

  force.distanceMax = function(_) {
    return arguments.length ? (distanceMax2 = _ * _, force) : Math.sqrt(distanceMax2);
  };

  force.theta = function(_) {
    return arguments.length ? (theta2 = _ * _, force) : Math.sqrt(theta2);
  };

  return force;
}