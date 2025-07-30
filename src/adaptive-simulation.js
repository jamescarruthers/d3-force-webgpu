import {forceSimulation as cpuSimulation} from "./simulation.js";
import {forceSimulationGPU} from "./webgpu/simulation-gpu.js";

// Performance thresholds based on typical benchmarks
const PERFORMANCE_THRESHOLDS = {
  // Below this node count, CPU is usually faster due to GPU overhead
  MIN_GPU_NODES: 750,
  
  // If CPU is significantly slower than this FPS, consider GPU
  MIN_CPU_FPS: 30,
  
  // If link count exceeds this, GPU becomes more beneficial
  MIN_GPU_LINKS: 1000,
  
  // Complexity score threshold (nodes * links)
  COMPLEXITY_THRESHOLD: 500000
};

// Auto-detection logic
function shouldUseGPU(nodes, estimatedLinkCount = nodes.length * 2) {
  const nodeCount = nodes.length;
  const complexity = nodeCount * estimatedLinkCount;
  
  // WebGPU not available
  if (!navigator.gpu) {
    return false;
  }
  
  // Too few nodes - GPU overhead not worth it
  if (nodeCount < PERFORMANCE_THRESHOLDS.MIN_GPU_NODES) {
    return false;
  }
  
  // High complexity - GPU likely beneficial
  if (complexity > PERFORMANCE_THRESHOLDS.COMPLEXITY_THRESHOLD) {
    return true;
  }
  
  // Medium complexity - GPU might be beneficial
  if (nodeCount >= PERFORMANCE_THRESHOLDS.MIN_GPU_NODES && 
      estimatedLinkCount >= PERFORMANCE_THRESHOLDS.MIN_GPU_LINKS) {
    return true;
  }
  
  return false;
}

// Performance monitor to switch implementations dynamically
class PerformanceMonitor {
  constructor() {
    this.fpsHistory = [];
    this.frameTimeHistory = [];
    this.lastTime = performance.now();
    this.frameCount = 0;
    this.avgFPS = 0;
    this.avgFrameTime = 0;
  }
  
  recordFrame(frameTime) {
    this.frameTimeHistory.push(frameTime);
    this.frameCount++;
    
    const now = performance.now();
    if (now - this.lastTime >= 1000) {
      const fps = Math.round(this.frameCount * 1000 / (now - this.lastTime));
      this.fpsHistory.push(fps);
      this.frameCount = 0;
      this.lastTime = now;
      
      // Calculate rolling averages
      const recentFPS = this.fpsHistory.slice(-5); // Last 5 seconds
      const recentFrameTimes = this.frameTimeHistory.slice(-300); // Last ~5 seconds at 60fps
      
      this.avgFPS = recentFPS.reduce((a, b) => a + b, 0) / recentFPS.length;
      this.avgFrameTime = recentFrameTimes.reduce((a, b) => a + b, 0) / recentFrameTimes.length;
    }
  }
  
  shouldSwitchToGPU() {
    return this.avgFPS < PERFORMANCE_THRESHOLDS.MIN_CPU_FPS && this.fpsHistory.length >= 3;
  }
  
  reset() {
    this.fpsHistory = [];
    this.frameTimeHistory = [];
    this.frameCount = 0;
    this.lastTime = performance.now();
  }
}

// Adaptive force simulation that chooses the best implementation
export default function forceSimulationAdaptive(nodes, options = {}) {
  const {
    mode = 'auto', // 'auto', 'cpu', 'gpu', 'hybrid'
    enableSwitching = true, // Allow dynamic switching based on performance
    onModeChange = null, // Callback when mode changes
    estimatedLinkCount = nodes?.length * 2,
    performanceThreshold = PERFORMANCE_THRESHOLDS.MIN_CPU_FPS
  } = options;
  
  let currentMode = mode;
  let currentSimulation = null;
  let monitor = new PerformanceMonitor();
  let switchCheckInterval = null;
  
  // Determine initial mode
  if (mode === 'auto') {
    currentMode = shouldUseGPU(nodes, estimatedLinkCount) ? 'gpu' : 'cpu';
  }
  
  async function createSimulation(nodes, useGPU = false) {
    if (useGPU) {
      try {
        const sim = forceSimulationGPU(nodes);
        await sim.initialize?.();
        return { simulation: sim, type: 'gpu', initialized: true };
      } catch (error) {
        console.warn('GPU simulation failed, falling back to CPU:', error.message);
        return { simulation: cpuSimulation(nodes), type: 'cpu', initialized: true };
      }
    } else {
      return { simulation: cpuSimulation(nodes), type: 'cpu', initialized: true };
    }
  }
  
  async function switchToGPU() {
    if (currentMode === 'gpu' || !enableSwitching) return;
    
    console.log('🚀 Switching to GPU acceleration for better performance');
    
    const oldSim = currentSimulation.simulation;
    const forces = new Map();
    
    // Preserve force configuration
    if (oldSim.forces) {
      oldSim.forces.forEach((force, name) => {
        forces.set(name, force);
      });
    }
    
    // Stop old simulation
    oldSim.stop();
    
    // Create new GPU simulation
    const newSim = await createSimulation(nodes, true);
    
    // Transfer forces
    forces.forEach((force, name) => {
      newSim.simulation.force(name, force);
    });
    
    // Update references
    currentSimulation = newSim;
    currentMode = 'gpu';
    monitor.reset();
    
    if (onModeChange) {
      onModeChange('gpu', 'Switched to GPU for better performance');
    }
    
    return newSim.simulation;
  }
  
  // Create initial simulation
  const initialSim = await createSimulation(nodes, currentMode === 'gpu');
  currentSimulation = initialSim;
  
  // Wrap simulation to add performance monitoring
  const originalTick = currentSimulation.simulation.tick;
  currentSimulation.simulation.tick = function(...args) {
    const startTime = performance.now();
    const result = originalTick.apply(this, args);
    const frameTime = performance.now() - startTime;
    
    monitor.recordFrame(frameTime);
    
    return result;
  };
  
  // Set up performance monitoring for dynamic switching
  if (enableSwitching && currentMode === 'cpu') {
    switchCheckInterval = setInterval(async () => {
      if (monitor.shouldSwitchToGPU()) {
        clearInterval(switchCheckInterval);
        await switchToGPU();
      }
    }, 2000); // Check every 2 seconds
  }
  
  // Extend simulation with adaptive methods
  const adaptiveSimulation = currentSimulation.simulation;
  
  // Add adaptive-specific methods
  adaptiveSimulation.getCurrentMode = () => currentMode;
  adaptiveSimulation.getPerformanceStats = () => ({
    avgFPS: monitor.avgFPS,
    avgFrameTime: monitor.avgFrameTime,
    mode: currentMode
  });
  
  adaptiveSimulation.forceMode = async (newMode) => {
    if (newMode === currentMode) return adaptiveSimulation;
    
    const oldSim = currentSimulation.simulation;
    const forces = new Map();
    
    // Preserve configuration
    if (oldSim.forces) {
      oldSim.forces.forEach((force, name) => {
        forces.set(name, force);
      });
    }
    
    oldSim.stop();
    
    // Create new simulation
    const newSim = await createSimulation(nodes, newMode === 'gpu');
    
    // Transfer forces
    forces.forEach((force, name) => {
      newSim.simulation.force(name, force);
    });
    
    currentSimulation = newSim;
    currentMode = newMode;
    monitor.reset();
    
    if (onModeChange) {
      onModeChange(newMode, `Manually switched to ${newMode.toUpperCase()}`);
    }
    
    return newSim.simulation;
  };
  
  // Override stop to clean up monitoring
  const originalStop = adaptiveSimulation.stop;
  adaptiveSimulation.stop = function() {
    if (switchCheckInterval) {
      clearInterval(switchCheckInterval);
      switchCheckInterval = null;
    }
    return originalStop.call(this);
  };
  
  // Add mode information to simulation
  adaptiveSimulation._adaptiveMode = currentMode;
  adaptiveSimulation._isAdaptive = true;
  
  if (onModeChange) {
    onModeChange(currentMode, `Started with ${currentMode.toUpperCase()} mode`);
  }
  
  return adaptiveSimulation;
}

// Export configuration utilities
export const AdaptiveConfig = {
  setThresholds: (newThresholds) => {
    Object.assign(PERFORMANCE_THRESHOLDS, newThresholds);
  },
  
  getThresholds: () => ({ ...PERFORMANCE_THRESHOLDS }),
  
  shouldUseGPU,
  
  estimateComplexity: (nodeCount, linkCount) => nodeCount * linkCount,
  
  recommendMode: (nodes, links = []) => {
    const complexity = nodes.length * links.length;
    
    if (!navigator.gpu) return 'cpu';
    if (nodes.length < 200) return 'cpu';
    if (complexity > 1000000) return 'gpu';
    if (nodes.length > 2000) return 'gpu';
    
    return 'auto';
  }
};

// Convenience exports for specific modes
export const forceCPU = cpuSimulation;
export const forceGPU = forceSimulationGPU;
export const forceAuto = forceSimulationAdaptive;