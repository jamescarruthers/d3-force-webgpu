export class BufferManager {
  constructor(device) {
    this.device = device;
    this.buffers = new Map();
  }

  createNodeBuffer(nodes) {
    const nodeCount = nodes.length;
    const floatsPerNode = 8; // x, y, vx, vy, fx, fy, index, _padding
    const byteSize = nodeCount * floatsPerNode * 4;

    const nodeData = new Float32Array(nodeCount * floatsPerNode);
    
    for (let i = 0; i < nodeCount; i++) {
      const node = nodes[i];
      const offset = i * floatsPerNode;
      
      nodeData[offset + 0] = node.x || 0;
      nodeData[offset + 1] = node.y || 0;
      nodeData[offset + 2] = node.vx || 0;
      nodeData[offset + 3] = node.vy || 0;
      nodeData[offset + 4] = node.fx !== null && node.fx !== undefined ? node.fx : NaN;
      nodeData[offset + 5] = node.fy !== null && node.fy !== undefined ? node.fy : NaN;
      nodeData[offset + 6] = i; // index
      nodeData[offset + 7] = 0; // padding for alignment
    }

    const buffer = this.device.createBuffer({
      label: 'Node Buffer',
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    new Float32Array(buffer.getMappedRange()).set(nodeData);
    buffer.unmap();

    this.buffers.set('nodes', {
      buffer,
      count: nodeCount,
      floatsPerNode
    });

    return buffer;
  }

  createSimulationParamsBuffer(params) {
    const paramData = new Float32Array([
      params.alpha || 1,
      params.alphaDecay || 0.0228,
      params.alphaTarget || 0,
      params.velocityDecay || 0.6,
      params.nodeCount || 0,
      0, // padding
      0, // padding
      0  // padding
    ]);

    const buffer = this.device.createBuffer({
      label: 'Simulation Parameters',
      size: paramData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    new Float32Array(buffer.getMappedRange()).set(paramData);
    buffer.unmap();

    this.buffers.set('simulationParams', { buffer });
    return buffer;
  }

  createReadbackBuffer(size) {
    const buffer = this.device.createBuffer({
      label: 'Readback Buffer',
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.buffers.set('readback', { buffer, size });
    return buffer;
  }

  async readNodeData(nodes) {
    const nodeBuffer = this.buffers.get('nodes');
    if (!nodeBuffer) return;

    const readbackBuffer = this.buffers.get('readback');
    if (!readbackBuffer || readbackBuffer.size < nodeBuffer.buffer.size) {
      this.createReadbackBuffer(nodeBuffer.buffer.size);
    }

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      nodeBuffer.buffer, 0,
      this.buffers.get('readback').buffer, 0,
      nodeBuffer.buffer.size
    );
    this.device.queue.submit([commandEncoder.finish()]);

    await this.buffers.get('readback').buffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.buffers.get('readback').buffer.getMappedRange());
    
    for (let i = 0; i < nodeBuffer.count; i++) {
      const offset = i * nodeBuffer.floatsPerNode;
      const node = nodes[i];
      
      node.x = data[offset + 0];
      node.y = data[offset + 1];
      node.vx = data[offset + 2];
      node.vy = data[offset + 3];
    }

    this.buffers.get('readback').buffer.unmap();
  }

  updateSimulationParams(params) {
    const paramData = new Float32Array([
      params.alpha,
      params.alphaDecay,
      params.alphaTarget,
      params.velocityDecay,
      params.nodeCount,
      0, 0, 0
    ]);

    this.device.queue.writeBuffer(
      this.buffers.get('simulationParams').buffer,
      0,
      paramData
    );
  }

  destroy() {
    for (const [, bufferInfo] of this.buffers) {
      bufferInfo.buffer.destroy();
    }
    this.buffers.clear();
  }
}