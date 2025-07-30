// Main adaptive d3-force package with intelligent CPU/GPU selection

// Export adaptive simulation (recommended)
export {default as forceSimulation, AdaptiveConfig, forceCPU, forceGPU, forceAuto} from "./adaptive-simulation.js";

// Export specific implementations for manual control
export {default as forceSimulationCPU} from "./simulation.js";
export {default as forceSimulationGPU} from "./webgpu/simulation-gpu.js";

// Export forces (work with both CPU and GPU)
export {default as forceCenter} from "./center.js";
export {default as forceCollide} from "./collide.js";
export {default as forceLink} from "./link.js";
export {default as forceManyBody} from "./manyBody.js";
export {default as forceRadial} from "./radial.js";
export {default as forceX} from "./x.js";
export {default as forceY} from "./y.js";

// Export GPU-specific forces
export {default as forceCenterGPU} from "./webgpu/forces/center-gpu.js";
export {default as forceManyBodyGPU} from "./webgpu/forces/manyBody-gpu.js";

// Utility functions
export function getRecommendedMode(nodes, links = []) {
  const nodeCount = nodes.length;
  const linkCount = links.length;
  const complexity = nodeCount * linkCount;
  
  if (!navigator.gpu) {
    return { mode: 'cpu', reason: 'WebGPU not supported' };
  }
  
  if (nodeCount < 200) {
    return { mode: 'cpu', reason: 'Small graph - CPU overhead lower' };
  }
  
  if (nodeCount < 750) {
    return { mode: 'cpu', reason: 'Medium graph - CPU likely competitive' };
  }
  
  if (nodeCount > 2000) {
    return { mode: 'gpu', reason: 'Large graph - GPU parallelism beneficial' };
  }
  
  if (complexity > 500000) {
    return { mode: 'gpu', reason: 'High complexity - GPU acceleration beneficial' };
  }
  
  return { mode: 'auto', reason: 'Let adaptive mode decide based on performance' };
}

export function createOptimalSimulation(nodes, options = {}) {
  const recommendation = getRecommendedMode(nodes, options.links);
  
  return forceSimulation(nodes, {
    mode: recommendation.mode,
    enableSwitching: true,
    onModeChange: (mode, reason) => {
      if (options.onModeChange) {
        options.onModeChange(mode, reason);
      } else {
        console.log(`🔄 D3-Force: ${reason}`);
      }
    },
    ...options
  });
}