// WebGPU Device Manager
// Handles device initialization, capability detection, and resource management

let gpuDevice = null;
let gpuAdapter = null;
let initPromise = null;

export async function initWebGPU() {
  if (gpuDevice) return gpuDevice;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser");
    }

    gpuAdapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });

    if (!gpuAdapter) {
      throw new Error("No WebGPU adapter found");
    }

    const requiredLimits = {
      maxStorageBufferBindingSize: gpuAdapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: gpuAdapter.limits.maxBufferSize,
      maxComputeWorkgroupsPerDimension: gpuAdapter.limits.maxComputeWorkgroupsPerDimension
    };

    gpuDevice = await gpuAdapter.requestDevice({
      requiredLimits
    });

    gpuDevice.lost.then((info) => {
      console.error("WebGPU device lost:", info.message);
      gpuDevice = null;
      gpuAdapter = null;
      initPromise = null;
    });

    // Handle uncaptured errors
    gpuDevice.onuncapturederror = (event) => {
      console.error("WebGPU uncaptured error:", event.error);
    };

    return gpuDevice;
  })();

  return initPromise;
}

export function getDevice() {
  return gpuDevice;
}

export function getAdapter() {
  return gpuAdapter;
}

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

export async function checkWebGPUSupport() {
  if (!isWebGPUAvailable()) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

// Utility to create a compute pipeline with error handling
export async function createComputePipeline(device, shaderCode, entryPoint, bindGroupLayout) {
  const shaderModule = device.createShaderModule({
    code: shaderCode
  });

  const compilationInfo = await shaderModule.getCompilationInfo();
  if (compilationInfo.messages.some(m => m.type === "error")) {
    const errors = compilationInfo.messages.filter(m => m.type === "error");
    throw new Error("Shader compilation failed: " + errors.map(e => e.message).join("\n"));
  }

  return device.createComputePipeline({
    layout: bindGroupLayout ? device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }) : "auto",
    compute: {
      module: shaderModule,
      entryPoint
    }
  });
}

// Utility to run a compute pass and wait for completion
export async function runComputePass(device, pipeline, bindGroup, workgroupCountX, workgroupCountY = 1, workgroupCountZ = 1) {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
}
