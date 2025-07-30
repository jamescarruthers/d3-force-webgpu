export default function(x, y) {
  var nodes,
      device,
      pipeline,
      bindGroup,
      centerParamsBuffer,
      strength = 1;

  if (x == null) x = 0;
  if (y == null) y = 0;

  async function force() {
    if (!device || !pipeline || !bindGroup) return;

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1); // Single workgroup for reduction
    passEncoder.end();
    
    device.queue.submit([commandEncoder.finish()]);
    
    // Wait for completion
    await device.queue.onSubmittedWorkDone();
  }

  force.initialize = function(_nodes, _random, _device, nodeBuffer) {
    nodes = _nodes;
    device = _device;

    const shaderCode = `
struct Node {
  position: vec2<f32>,
  velocity: vec2<f32>,
  fixedPosition: vec2<f32>,
  index: f32,
  _padding: f32,
}

struct CenterParams {
  center: vec2<f32>,
  strength: f32,
  nodeCount: f32,
}

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<uniform> params: CenterParams;
@group(0) @binding(2) var<storage, read_write> reduction: array<vec2<f32>>;

var<workgroup> shared_sum: array<vec2<f32>, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let tid = local_id.x;
  let gid = global_id.x;
  let nodeCount = u32(params.nodeCount);
  
  // First pass: compute center of mass
  if (workgroup_id.x == 0u) {
    var sum = vec2<f32>(0.0, 0.0);
    
    // Each thread sums multiple nodes
    for (var i = gid; i < nodeCount; i += 64u) {
      sum += nodes[i].position;
    }
    
    shared_sum[tid] = sum;
    workgroupBarrier();
    
    // Reduction in shared memory
    for (var s = 32u; s > 0u; s >>= 1u) {
      if (tid < s) {
        shared_sum[tid] += shared_sum[tid + s];
      }
      workgroupBarrier();
    }
    
    // Write result
    if (tid == 0u) {
      reduction[0] = shared_sum[0] / f32(nodeCount);
    }
  }
  
  workgroupBarrier();
  storageBarrier();
  
  // Second pass: apply centering force
  if (gid < nodeCount) {
    let center_of_mass = reduction[0];
    let delta = (params.center - center_of_mass) * params.strength;
    nodes[gid].position += delta;
  }
}`;

    const shaderModule = device.createShaderModule({
      label: 'Center Force Shader',
      code: shaderCode
    });

    centerParamsBuffer = device.createBuffer({
      label: 'Center Parameters',
      size: 4 * 4, // vec2 + 2 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    
    new Float32Array(centerParamsBuffer.getMappedRange()).set([
      x, y, strength, nodes.length
    ]);
    centerParamsBuffer.unmap();

    const reductionBuffer = device.createBuffer({
      label: 'Reduction Buffer',
      size: 8, // vec2
      usage: GPUBufferUsage.STORAGE
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
          buffer: { type: 'uniform' }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
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
        { binding: 1, resource: { buffer: centerParamsBuffer } },
        { binding: 2, resource: { buffer: reductionBuffer } }
      ]
    });
  };

  force.x = function(_) {
    return arguments.length ? (x = +_, updateCenterParams(), force) : x;
  };

  force.y = function(_) {
    return arguments.length ? (y = +_, updateCenterParams(), force) : y;
  };

  force.strength = function(_) {
    return arguments.length ? (strength = +_, updateCenterParams(), force) : strength;
  };

  function updateCenterParams() {
    if (device && centerParamsBuffer && nodes) {
      const params = new Float32Array([x, y, strength, nodes.length]);
      device.queue.writeBuffer(centerParamsBuffer, 0, params);
    }
  }

  return force;
}