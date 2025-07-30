export class WebGPUContext {
  constructor() {
    this.device = null;
    this.adapter = null;
  }

  async initialize() {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser');
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });

    if (!this.adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    this.device = await this.adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {
        maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: this.adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup,
      }
    });

    this.device.lost.then((info) => {
      console.error('WebGPU device was lost:', info.message);
      if (info.reason !== 'destroyed') {
        this.initialize();
      }
    });

    return this.device;
  }

  destroy() {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
  }
}

let sharedContext = null;

export async function getWebGPUContext() {
  if (!sharedContext) {
    sharedContext = new WebGPUContext();
    await sharedContext.initialize();
  }
  return sharedContext;
}