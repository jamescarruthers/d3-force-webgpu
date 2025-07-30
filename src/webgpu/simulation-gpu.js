import {dispatch} from "d3-dispatch";
import {timer} from "d3-timer";
import lcg from "../lcg.js";
import {getWebGPUContext} from "./webgpu-context.js";
import {BufferManager} from "./buffer-manager.js";
import {ShaderLoader} from "./shader-loader.js";

export function x(d) {
  return d.x;
}

export function y(d) {
  return d.y;
}

var initialRadius = 10,
    initialAngle = Math.PI * (3 - Math.sqrt(5));

export default function(nodes) {
  var simulation,
      alpha = 1,
      alphaMin = 0.001,
      alphaDecay = 1 - Math.pow(alphaMin, 1 / 300),
      alphaTarget = 0,
      velocityDecay = 0.6,
      forces = new Map(),
      stepper = timer(step),
      event = dispatch("tick", "end"),
      random = lcg(),
      device = null,
      bufferManager = null,
      shaderLoader = null,
      simulationPipeline = null,
      bindGroup = null,
      isGPUInitialized = false;

  if (nodes == null) nodes = [];

  async function initializeGPU() {
    if (isGPUInitialized) return;

    try {
      const context = await getWebGPUContext();
      device = context.device;
      
      bufferManager = new BufferManager(device);
      shaderLoader = new ShaderLoader(device);
      
      const {pipeline, bindGroupLayout} = shaderLoader.createSimulationTickPipeline();
      simulationPipeline = pipeline;
      
      const nodeBuffer = bufferManager.createNodeBuffer(nodes);
      const paramsBuffer = bufferManager.createSimulationParamsBuffer({
        alpha,
        alphaDecay,
        alphaTarget,
        velocityDecay,
        nodeCount: nodes.length
      });
      
      bindGroup = shaderLoader.createBindGroup('simulationTick', bindGroupLayout, [
        nodeBuffer,
        paramsBuffer
      ]);
      
      isGPUInitialized = true;
    } catch (error) {
      console.error('Failed to initialize WebGPU:', error);
      throw error;
    }
  }

  async function step() {
    await tick();
    event.call("tick", simulation);
    if (alpha < alphaMin) {
      stepper.stop();
      event.call("end", simulation);
    }
  }

  async function tick(iterations) {
    if (iterations === undefined) iterations = 1;

    if (!isGPUInitialized) {
      await initializeGPU();
    }

    for (var k = 0; k < iterations; ++k) {
      alpha += (alphaTarget - alpha) * alphaDecay;

      bufferManager.updateSimulationParams({
        alpha,
        alphaDecay,
        alphaTarget,
        velocityDecay,
        nodeCount: nodes.length
      });

      // Apply forces first (they should update velocities on GPU)
      for (const force of forces.values()) {
        if (force && typeof force === 'function') {
          await force(alpha);
        }
      }

      // Then apply velocity integration
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      
      passEncoder.setPipeline(simulationPipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(nodes.length / 64));
      passEncoder.end();
      
      device.queue.submit([commandEncoder.finish()]);

      // Wait for GPU to finish
      await device.queue.onSubmittedWorkDone();

      // Read back the updated node positions
      await bufferManager.readNodeData(nodes);
    }

    return simulation;
  }

  function initializeNodes() {
    for (var i = 0, n = nodes.length, node; i < n; ++i) {
      node = nodes[i], node.index = i;
      if (node.fx != null) node.x = node.fx;
      if (node.fy != null) node.y = node.fy;
      if (isNaN(node.x) || isNaN(node.y)) {
        var radius = initialRadius * Math.sqrt(0.5 + i), angle = i * initialAngle;
        node.x = radius * Math.cos(angle);
        node.y = radius * Math.sin(angle);
      }
      if (isNaN(node.vx) || isNaN(node.vy)) {
        node.vx = node.vy = 0;
      }
    }
  }

  function initializeForce(force) {
    if (force.initialize) {
      const nodeBuffer = bufferManager ? bufferManager.buffers.get('nodes')?.buffer : null;
      force.initialize(nodes, random, device, nodeBuffer);
    }
    return force;
  }

  initializeNodes();

  return simulation = {
    tick: tick,

    restart: function() {
      return stepper.restart(step), simulation;
    },

    stop: function() {
      return stepper.stop(), simulation;
    },

    nodes: function(_) {
      if (!arguments.length) return nodes;
      nodes = _;
      initializeNodes();
      forces.forEach(initializeForce);
      if (isGPUInitialized) {
        bufferManager.destroy();
        isGPUInitialized = false;
      }
      return simulation;
    },

    alpha: function(_) {
      return arguments.length ? (alpha = +_, simulation) : alpha;
    },

    alphaMin: function(_) {
      return arguments.length ? (alphaMin = +_, simulation) : alphaMin;
    },

    alphaDecay: function(_) {
      return arguments.length ? (alphaDecay = +_, simulation) : +alphaDecay;
    },

    alphaTarget: function(_) {
      return arguments.length ? (alphaTarget = +_, simulation) : alphaTarget;
    },

    velocityDecay: function(_) {
      return arguments.length ? (velocityDecay = 1 - _, simulation) : 1 - velocityDecay;
    },

    randomSource: function(_) {
      return arguments.length ? (random = _, forces.forEach(initializeForce), simulation) : random;
    },

    force: function(name, _) {
      return arguments.length > 1 ? ((_ == null ? forces.delete(name) : forces.set(name, initializeForce(_))), simulation) : forces.get(name);
    },

    find: function(x, y, radius) {
      var i = 0,
          n = nodes.length,
          dx,
          dy,
          d2,
          node,
          closest;

      if (radius == null) radius = Infinity;
      else radius *= radius;

      for (i = 0; i < n; ++i) {
        node = nodes[i];
        dx = x - node.x;
        dy = y - node.y;
        d2 = dx * dx + dy * dy;
        if (d2 < radius) closest = node, radius = d2;
      }

      return closest;
    },

    on: function(name, _) {
      return arguments.length > 1 ? (event.on(name, _), simulation) : event.on(name);
    },

    destroy: function() {
      stepper.stop();
      if (bufferManager) bufferManager.destroy();
      if (shaderLoader) shaderLoader.destroy();
      isGPUInitialized = false;
    }
  };
}