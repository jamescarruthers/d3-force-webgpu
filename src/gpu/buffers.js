// GPU Buffer Manager
// Handles creation and synchronization of GPU buffers for node/link data

export class NodeBuffers {
  constructor(device, nodeCount) {
    this.device = device;
    this.nodeCount = nodeCount;
    this.isMapped = false;

    // Each node has: x, y, vx, vy (4 floats = 16 bytes)
    // We also need fx, fy for fixed positions (use NaN to indicate not fixed)
    // And strength for many-body force
    // Layout: [x, y, vx, vy, fx, fy, strength, radius] = 8 floats per node = 32 bytes
    this.floatsPerNode = 8;
    this.bytesPerNode = this.floatsPerNode * 4;
    this.bufferSize = this.nodeCount * this.bytesPerNode;

    // Main storage buffer (GPU read/write)
    this.storageBuffer = device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: "Node Storage Buffer"
    });

    // Staging buffer for reading back to CPU
    this.stagingBuffer = device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: "Node Staging Buffer"
    });

    // CPU-side typed array for uploads
    this.cpuData = new Float32Array(this.nodeCount * this.floatsPerNode);
  }

  // Upload node data from JS objects to GPU
  uploadNodes(nodes) {
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const offset = i * this.floatsPerNode;
      this.cpuData[offset + 0] = node.x || 0;
      this.cpuData[offset + 1] = node.y || 0;
      this.cpuData[offset + 2] = node.vx || 0;
      this.cpuData[offset + 3] = node.vy || 0;
      this.cpuData[offset + 4] = node.fx != null ? node.fx : NaN;
      this.cpuData[offset + 5] = node.fy != null ? node.fy : NaN;
      this.cpuData[offset + 6] = node._strength != null ? node._strength : -30;
      this.cpuData[offset + 7] = node.radius != null ? node.radius : 5;
    }
    this.device.queue.writeBuffer(this.storageBuffer, 0, this.cpuData);
  }

  // Read node positions back from GPU to JS objects
  async downloadNodes(nodes) {
    // Safety check - don't map if already mapped
    if (this.isMapped) {
      console.warn("Buffer already mapped, skipping download");
      return;
    }

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.storageBuffer, 0, this.stagingBuffer, 0, this.bufferSize);
    this.device.queue.submit([commandEncoder.finish()]);

    this.isMapped = true;
    try {
      await this.stagingBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(this.stagingBuffer.getMappedRange().slice(0));
      this.stagingBuffer.unmap();

      for (let i = 0; i < nodes.length; i++) {
        const offset = i * this.floatsPerNode;
        nodes[i].x = data[offset + 0];
        nodes[i].y = data[offset + 1];
        nodes[i].vx = data[offset + 2];
        nodes[i].vy = data[offset + 3];
      }
    } finally {
      this.isMapped = false;
    }
  }

  // Update velocities only (after force computation)
  async downloadVelocities(nodes) {
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.storageBuffer, 0, this.stagingBuffer, 0, this.bufferSize);
    this.device.queue.submit([commandEncoder.finish()]);

    await this.stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.stagingBuffer.getMappedRange().slice(0));
    this.stagingBuffer.unmap();

    for (let i = 0; i < nodes.length; i++) {
      const offset = i * this.floatsPerNode;
      nodes[i].vx = data[offset + 2];
      nodes[i].vy = data[offset + 3];
    }
  }

  destroy() {
    this.storageBuffer.destroy();
    this.stagingBuffer.destroy();
  }
}

export class LinkBuffers {
  constructor(device, linkCount) {
    this.device = device;
    this.linkCount = linkCount;

    // Each link has: sourceIndex, targetIndex, distance, strength, bias (5 values)
    // Use 8 floats for alignment: [sourceIdx, targetIdx, distance, strength, bias, pad, pad, pad]
    this.floatsPerLink = 8;
    this.bytesPerLink = this.floatsPerLink * 4;
    this.bufferSize = Math.max(32, this.linkCount * this.bytesPerLink);

    this.storageBuffer = device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "Link Storage Buffer"
    });

    this.cpuData = new Float32Array(this.linkCount * this.floatsPerLink);
  }

  uploadLinks(links, distances, strengths, bias) {
    for (let i = 0; i < links.length; i++) {
      const link = links[i];
      const offset = i * this.floatsPerLink;
      this.cpuData[offset + 0] = link.source.index;
      this.cpuData[offset + 1] = link.target.index;
      this.cpuData[offset + 2] = distances[i];
      this.cpuData[offset + 3] = strengths[i];
      this.cpuData[offset + 4] = bias[i];
      // padding
      this.cpuData[offset + 5] = 0;
      this.cpuData[offset + 6] = 0;
      this.cpuData[offset + 7] = 0;
    }
    this.device.queue.writeBuffer(this.storageBuffer, 0, this.cpuData);
  }

  destroy() {
    this.storageBuffer.destroy();
  }
}

export class SimulationParamsBuffer {
  constructor(device) {
    this.device = device;

    // Simulation parameters matching shader struct (24 floats = 96 bytes):
    // 0: alpha: f32, 1: velocityDecay: f32, 2: nodeCount: u32, 3: linkCount: u32,
    // 4: centerX: f32, 5: centerY: f32, 6: centerStrength: f32, 7: theta2: f32,
    // 8: distanceMin2: f32, 9: distanceMax2: f32, 10: iterations: u32, 11: collisionRadius: f32,
    // 12: collisionStrength: f32, 13: collisionIterations: u32, 14: forceXTarget: f32, 15: forceXStrength: f32,
    // 16: forceYTarget: f32, 17: forceYStrength: f32, 18: radialX: f32, 19: radialY: f32,
    // 20: radialRadius: f32, 21: radialStrength: f32, 22: _pad1: f32, 23: _pad2: f32
    // 24 x 4 bytes = 96 bytes (aligned to 16 bytes)
    this.bufferSize = 96;

    this.storageBuffer = device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "Simulation Params Buffer"
    });

    // Use ArrayBuffer with views to properly handle mixed f32/u32 types
    this.arrayBuffer = new ArrayBuffer(96);
    this.floatView = new Float32Array(this.arrayBuffer);
    this.uintView = new Uint32Array(this.arrayBuffer);
  }

  update(params) {
    // f32 values
    this.floatView[0] = params.alpha || 1;
    this.floatView[1] = params.velocityDecay || 0.6;
    // u32 values (indices 2, 3)
    this.uintView[2] = params.nodeCount || 0;
    this.uintView[3] = params.linkCount || 0;
    // f32 values
    this.floatView[4] = params.centerX || 0;
    this.floatView[5] = params.centerY || 0;
    this.floatView[6] = params.centerStrength || 1;
    this.floatView[7] = params.theta2 || 0.81;
    this.floatView[8] = params.distanceMin2 || 1;
    // Use a very large finite number instead of Infinity for GPU compatibility
    const maxDist2 = params.distanceMax2;
    this.floatView[9] = (maxDist2 === Infinity || maxDist2 > 1e30) ? 1e30 : maxDist2;
    // u32 value
    this.uintView[10] = params.iterations || 1;
    // f32 value
    this.floatView[11] = params.collisionRadius || 5;
    this.floatView[12] = params.collisionStrength || 1;
    // u32 value
    this.uintView[13] = params.collisionIterations || 1;
    // forceX params
    this.floatView[14] = params.forceXTarget || 0;
    this.floatView[15] = params.forceXStrength || 0.1;
    // forceY params
    this.floatView[16] = params.forceYTarget || 0;
    this.floatView[17] = params.forceYStrength || 0.1;
    // radial params
    this.floatView[18] = params.radialX || 0;
    this.floatView[19] = params.radialY || 0;
    this.floatView[20] = params.radialRadius || 100;
    this.floatView[21] = params.radialStrength || 0.1;
    // padding
    this.floatView[22] = 0;
    this.floatView[23] = 0;

    this.device.queue.writeBuffer(this.storageBuffer, 0, this.arrayBuffer);
  }

  destroy() {
    this.storageBuffer.destroy();
  }
}
