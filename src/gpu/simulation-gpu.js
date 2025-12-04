// GPU-Accelerated Force Simulation
// Drop-in replacement for forceSimulation with WebGPU acceleration

import { dispatch } from "d3-dispatch";
import { timer } from "d3-timer";
import { initWebGPU, getDevice, createComputePipeline } from "./device.js";
import { NodeBuffers, LinkBuffers, SimulationParamsBuffer } from "./buffers.js";
import { manyBodyShader, linkShader, integrateShader, collisionShader, forceXShader, forceYShader, forceRadialShader } from "./shaders.js";
import lcg from "../lcg.js";

const WORKGROUP_SIZE = 256;

export function x(d) {
  return d.x;
}

export function y(d) {
  return d.y;
}

var initialRadius = 10,
  initialAngle = Math.PI * (3 - Math.sqrt(5));

export default function (nodes) {
  var simulation,
    alpha = 1,
    alphaMin = 0.001,
    alphaDecay = 1 - Math.pow(alphaMin, 1 / 300),
    alphaTarget = 0,
    velocityDecay = 0.6,
    forces = new Map(),
    stepper = timer(step),
    event = dispatch("tick", "end"),
    random = lcg();

  // GPU resources
  var gpuInitialized = false,
    gpuDevice = null,
    nodeBuffers = null,
    paramsBuffer = null,
    pipelines = {},
    bindGroups = {},
    gpuTickInProgress = false,
    gpuReadyPromise = null,
    gpuReadyResolve = null,
    gpuLoopRunning = false; // GPU compute loop running flag

  // Force-specific GPU resources
  var linkBuffers = null,
    linkData = { links: [], distances: [], strengths: [], bias: [] };

  // Force configurations (mirroring CPU forces)
  var manyBodyConfig = {
    enabled: false,
    strength: -30,
    theta: 0.9,
    distanceMin: 1,
    distanceMax: Infinity,
  };

  var centerConfig = {
    enabled: false,
    x: 0,
    y: 0,
    strength: 1,
  };

  var collisionConfig = {
    enabled: false,
    radius: 5,
    strength: 1,
    iterations: 1,
  };

  var forceXConfig = {
    enabled: false,
    x: 0,
    strength: 0.1,
  };

  var forceYConfig = {
    enabled: false,
    y: 0,
    strength: 0.1,
  };

  var radialConfig = {
    enabled: false,
    x: 0,
    y: 0,
    radius: 100,
    strength: 0.1,
  };

  if (nodes == null) nodes = [];

  // Create promise for gpuReady()
  gpuReadyPromise = new Promise((resolve) => {
    gpuReadyResolve = resolve;
  });

  async function initGPU() {
    if (gpuInitialized) return true;

    try {
      gpuDevice = await initWebGPU();
      await createPipelines();
      gpuInitialized = true;
      return true;
    } catch (e) {
      console.warn("WebGPU initialization failed, falling back to CPU:", e);
      return false;
    }
  }

  async function createPipelines() {
    // Many-body force pipeline
    pipelines.manyBody = await createComputePipeline(
      gpuDevice,
      manyBodyShader,
      "main"
    );

    // Link force pipeline
    pipelines.linkForce = await createComputePipeline(
      gpuDevice,
      linkShader,
      "main"
    );

    // Integration pipeline
    pipelines.integrate = await createComputePipeline(
      gpuDevice,
      integrateShader,
      "main"
    );

    // Collision pipeline
    pipelines.collision = await createComputePipeline(
      gpuDevice,
      collisionShader,
      "main"
    );

    // ForceX pipeline
    pipelines.forceX = await createComputePipeline(
      gpuDevice,
      forceXShader,
      "main"
    );

    // ForceY pipeline
    pipelines.forceY = await createComputePipeline(
      gpuDevice,
      forceYShader,
      "main"
    );

    // Radial force pipeline
    pipelines.radial = await createComputePipeline(
      gpuDevice,
      forceRadialShader,
      "main"
    );
  }

  function createBuffers() {
    if (!gpuDevice || nodes.length === 0) return;

    // Destroy old buffers
    if (nodeBuffers) nodeBuffers.destroy();
    if (paramsBuffer) paramsBuffer.destroy();
    if (linkBuffers) linkBuffers.destroy();

    nodeBuffers = new NodeBuffers(gpuDevice, nodes.length);
    paramsBuffer = new SimulationParamsBuffer(gpuDevice);

    // Create link buffers if we have links
    if (linkData.links.length > 0) {
      linkBuffers = new LinkBuffers(gpuDevice, linkData.links.length);
    }

    createBindGroups();
  }

  function createBindGroups() {
    if (!gpuDevice || !nodeBuffers) return;

    // Many-body bind group
    bindGroups.manyBody = gpuDevice.createBindGroup({
      layout: pipelines.manyBody.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.storageBuffer } },
      ],
    });

    // Integration bind group
    bindGroups.integrate = gpuDevice.createBindGroup({
      layout: pipelines.integrate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.storageBuffer } },
      ],
    });

    // Collision bind group
    bindGroups.collision = gpuDevice.createBindGroup({
      layout: pipelines.collision.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.storageBuffer } },
      ],
    });

    // Link bind group (if links exist)
    if (linkBuffers) {
      bindGroups.link = gpuDevice.createBindGroup({
        layout: pipelines.linkForce.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
          { binding: 1, resource: { buffer: linkBuffers.storageBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer.storageBuffer } },
        ],
      });
    }

    // ForceX bind group
    bindGroups.forceX = gpuDevice.createBindGroup({
      layout: pipelines.forceX.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.storageBuffer } },
      ],
    });

    // ForceY bind group
    bindGroups.forceY = gpuDevice.createBindGroup({
      layout: pipelines.forceY.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.storageBuffer } },
      ],
    });

    // Radial force bind group
    bindGroups.radial = gpuDevice.createBindGroup({
      layout: pipelines.radial.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffers.storageBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.storageBuffer } },
      ],
    });
  }

  function uploadToGPU() {
    if (!nodeBuffers) return;

    // Set per-node strengths for many-body force
    for (let i = 0; i < nodes.length; i++) {
      nodes[i]._strength = manyBodyConfig.strength;
      nodes[i].radius = nodes[i].radius || collisionConfig.radius;
    }

    nodeBuffers.uploadNodes(nodes);

    paramsBuffer.update({
      alpha: alpha,
      velocityDecay: velocityDecay,
      nodeCount: nodes.length,
      linkCount: linkData.links.length,
      centerX: centerConfig.x,
      centerY: centerConfig.y,
      centerStrength: centerConfig.strength,
      theta2: manyBodyConfig.theta * manyBodyConfig.theta,
      distanceMin2: manyBodyConfig.distanceMin * manyBodyConfig.distanceMin,
      distanceMax2: manyBodyConfig.distanceMax * manyBodyConfig.distanceMax,
      iterations: 1,
      collisionRadius: collisionConfig.radius,
      collisionStrength: collisionConfig.strength,
      collisionIterations: collisionConfig.iterations,
      forceXTarget: forceXConfig.x,
      forceXStrength: forceXConfig.enabled ? forceXConfig.strength : 0,
      forceYTarget: forceYConfig.y,
      forceYStrength: forceYConfig.enabled ? forceYConfig.strength : 0,
      radialX: radialConfig.x,
      radialY: radialConfig.y,
      radialRadius: radialConfig.radius,
      radialStrength: radialConfig.enabled ? radialConfig.strength : 0,
    });

    if (linkBuffers && linkData.links.length > 0) {
      linkBuffers.uploadLinks(
        linkData.links,
        linkData.distances,
        linkData.strengths,
        linkData.bias
      );
    }
  }

  // Run a single GPU tick (compute + readback)
  async function runGPUTick() {
    alpha += (alphaTarget - alpha) * alphaDecay;

    uploadToGPU();

    const workgroups = Math.ceil(nodes.length / WORKGROUP_SIZE);
    const commandEncoder = gpuDevice.createCommandEncoder();

    // Apply forces in order
    const passEncoder = commandEncoder.beginComputePass();

    // 1. Many-body force
    if (manyBodyConfig.enabled && bindGroups.manyBody) {
      passEncoder.setPipeline(pipelines.manyBody);
      passEncoder.setBindGroup(0, bindGroups.manyBody);
      passEncoder.dispatchWorkgroups(workgroups);
    }

    // 2. Link force
    if (linkData.links.length > 0 && bindGroups.link) {
      passEncoder.setPipeline(pipelines.linkForce);
      passEncoder.setBindGroup(0, bindGroups.link);
      passEncoder.dispatchWorkgroups(workgroups);
    }

    // 3. Collision force
    if (collisionConfig.enabled && bindGroups.collision) {
      for (let ci = 0; ci < collisionConfig.iterations; ci++) {
        passEncoder.setPipeline(pipelines.collision);
        passEncoder.setBindGroup(0, bindGroups.collision);
        passEncoder.dispatchWorkgroups(workgroups);
      }
    }

    // 4. ForceX (x-positioning force)
    if (forceXConfig.enabled && bindGroups.forceX) {
      passEncoder.setPipeline(pipelines.forceX);
      passEncoder.setBindGroup(0, bindGroups.forceX);
      passEncoder.dispatchWorkgroups(workgroups);
    }

    // 5. ForceY (y-positioning force)
    if (forceYConfig.enabled && bindGroups.forceY) {
      passEncoder.setPipeline(pipelines.forceY);
      passEncoder.setBindGroup(0, bindGroups.forceY);
      passEncoder.dispatchWorkgroups(workgroups);
    }

    // 6. Radial force
    if (radialConfig.enabled && bindGroups.radial) {
      passEncoder.setPipeline(pipelines.radial);
      passEncoder.setBindGroup(0, bindGroups.radial);
      passEncoder.dispatchWorkgroups(workgroups);
    }

    // 7. Integration (position update)
    passEncoder.setPipeline(pipelines.integrate);
    passEncoder.setBindGroup(0, bindGroups.integrate);
    passEncoder.dispatchWorkgroups(workgroups);

    passEncoder.end();
    gpuDevice.queue.submit([commandEncoder.finish()]);

    // Read back results
    await nodeBuffers.downloadNodes(nodes);

    // Apply center force on CPU (requires reduction)
    if (centerConfig.enabled) {
      applyCenterCPU();
    }
  }

  // GPU compute loop - runs independently of d3-timer
  async function gpuComputeLoop() {
    if (gpuLoopRunning) return;
    gpuLoopRunning = true;

    try {
      while (alpha >= alphaMin && gpuLoopRunning && gpuInitialized) {
        await runGPUTick();
        // Fire tick event after each GPU computation completes
        event.call("tick", simulation);
      }

      if (alpha < alphaMin) {
        gpuLoopRunning = false;
        event.call("end", simulation);
      }
    } catch (e) {
      console.error("GPU compute loop failed, falling back to CPU:", e);
      gpuInitialized = false;
      gpuLoopRunning = false;
    }
  }

  async function tickGPU(iterations) {
    if (!gpuInitialized || !nodeBuffers || !gpuDevice) {
      // Fallback to CPU
      return tickCPU(iterations);
    }

    // GPU mode: the compute loop handles ticks, not the timer
    // Just start the loop if not already running
    if (!gpuLoopRunning) {
      gpuComputeLoop();
    }

    return simulation;
  }

  function applyCenterCPU() {
    let sx = 0,
      sy = 0;
    for (let i = 0; i < nodes.length; i++) {
      sx += nodes[i].x;
      sy += nodes[i].y;
    }
    sx = (sx / nodes.length - centerConfig.x) * centerConfig.strength;
    sy = (sy / nodes.length - centerConfig.y) * centerConfig.strength;
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].x -= sx;
      nodes[i].y -= sy;
    }
  }

  function tickCPU(iterations) {
    var i,
      n = nodes.length,
      node;

    if (iterations === undefined) iterations = 1;

    for (var k = 0; k < iterations; ++k) {
      alpha += (alphaTarget - alpha) * alphaDecay;

      forces.forEach(function (force) {
        force(alpha);
      });

      for (i = 0; i < n; ++i) {
        node = nodes[i];
        if (node.fx == null) node.x += node.vx *= velocityDecay;
        else (node.x = node.fx), (node.vx = 0);
        if (node.fy == null) node.y += node.vy *= velocityDecay;
        else (node.y = node.fy), (node.vy = 0);
      }
    }

    return simulation;
  }

  function step() {
    tick();
    // In GPU mode, tick events are fired by the GPU compute loop
    // In CPU mode, we fire them here
    if (!gpuInitialized) {
      event.call("tick", simulation);
      if (alpha < alphaMin) {
        stepper.stop();
        event.call("end", simulation);
      }
    }
  }

  function tick(iterations) {
    if (gpuInitialized) {
      tickGPU(iterations);
    } else {
      tickCPU(iterations);
    }
    return simulation;
  }

  function initializeNodes() {
    for (var i = 0, n = nodes.length, node; i < n; ++i) {
      (node = nodes[i]), (node.index = i);
      if (node.fx != null) node.x = node.fx;
      if (node.fy != null) node.y = node.fy;
      if (isNaN(node.x) || isNaN(node.y)) {
        var radius = initialRadius * Math.sqrt(0.5 + i),
          angle = i * initialAngle;
        node.x = radius * Math.cos(angle);
        node.y = radius * Math.sin(angle);
      }
      if (isNaN(node.vx) || isNaN(node.vy)) {
        node.vx = node.vy = 0;
      }
    }
  }

  function initializeForce(force) {
    if (force.initialize) force.initialize(nodes, random);
    return force;
  }

  initializeNodes();

  // Warmup pass to force shader compilation
  async function warmupGPU() {
    if (!gpuDevice || !nodeBuffers || nodes.length === 0) return;

    // Run a single compute pass to trigger shader compilation
    const workgroups = Math.ceil(nodes.length / WORKGROUP_SIZE);
    const commandEncoder = gpuDevice.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    // Dispatch each pipeline once to compile shaders
    if (bindGroups.manyBody) {
      passEncoder.setPipeline(pipelines.manyBody);
      passEncoder.setBindGroup(0, bindGroups.manyBody);
      passEncoder.dispatchWorkgroups(workgroups);
    }
    if (bindGroups.collision) {
      passEncoder.setPipeline(pipelines.collision);
      passEncoder.setBindGroup(0, bindGroups.collision);
      passEncoder.dispatchWorkgroups(workgroups);
    }
    if (bindGroups.forceX) {
      passEncoder.setPipeline(pipelines.forceX);
      passEncoder.setBindGroup(0, bindGroups.forceX);
      passEncoder.dispatchWorkgroups(workgroups);
    }
    if (bindGroups.forceY) {
      passEncoder.setPipeline(pipelines.forceY);
      passEncoder.setBindGroup(0, bindGroups.forceY);
      passEncoder.dispatchWorkgroups(workgroups);
    }
    if (bindGroups.radial) {
      passEncoder.setPipeline(pipelines.radial);
      passEncoder.setBindGroup(0, bindGroups.radial);
      passEncoder.dispatchWorkgroups(workgroups);
    }
    passEncoder.setPipeline(pipelines.integrate);
    passEncoder.setBindGroup(0, bindGroups.integrate);
    passEncoder.dispatchWorkgroups(workgroups);

    passEncoder.end();
    gpuDevice.queue.submit([commandEncoder.finish()]);

    // Wait for GPU to finish compilation
    await gpuDevice.queue.onSubmittedWorkDone();
  }

  // Start GPU initialization
  initGPU().then(async () => {
    if (gpuInitialized) {
      createBuffers();
      await warmupGPU();
    }
    gpuReadyResolve(gpuInitialized);
  });

  return (simulation = {
    tick: tick,

    restart: function () {
      stepper.restart(step);
      // Restart GPU compute loop if in GPU mode
      if (gpuInitialized && !gpuLoopRunning) {
        gpuComputeLoop();
      }
      return simulation;
    },

    stop: function () {
      gpuLoopRunning = false; // Stop GPU compute loop
      return stepper.stop(), simulation;
    },

    nodes: function (_) {
      if (!arguments.length) return nodes;
      nodes = _;
      initializeNodes();
      forces.forEach(initializeForce);
      if (gpuInitialized) createBuffers();
      return simulation;
    },

    alpha: function (_) {
      return arguments.length ? ((alpha = +_), simulation) : alpha;
    },

    alphaMin: function (_) {
      return arguments.length ? ((alphaMin = +_), simulation) : alphaMin;
    },

    alphaDecay: function (_) {
      return arguments.length ? ((alphaDecay = +_), simulation) : +alphaDecay;
    },

    alphaTarget: function (_) {
      return arguments.length ? ((alphaTarget = +_), simulation) : alphaTarget;
    },

    velocityDecay: function (_) {
      return arguments.length
        ? ((velocityDecay = 1 - _), simulation)
        : 1 - velocityDecay;
    },

    randomSource: function (_) {
      return arguments.length
        ? ((random = _), forces.forEach(initializeForce), simulation)
        : random;
    },

    force: function (name, _) {
      if (arguments.length > 1) {
        if (_ == null) {
          forces.delete(name);
          // Disable GPU force
          if (name === "charge" || name === "manyBody") {
            manyBodyConfig.enabled = false;
          } else if (name === "center") {
            centerConfig.enabled = false;
          } else if (name === "collide") {
            collisionConfig.enabled = false;
          } else if (name === "link") {
            linkData = { links: [], distances: [], strengths: [], bias: [] };
            if (linkBuffers) {
              linkBuffers.destroy();
              linkBuffers = null;
            }
          } else if (name === "x") {
            forceXConfig.enabled = false;
          } else if (name === "y") {
            forceYConfig.enabled = false;
          } else if (name === "radial") {
            radialConfig.enabled = false;
          }
        } else {
          forces.set(name, initializeForce(_));
          // Configure GPU force based on CPU force settings
          configureGPUForce(name, _);
        }
        return simulation;
      }
      return forces.get(name);
    },

    find: function (x, y, radius) {
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
        if (d2 < radius) (closest = node), (radius = d2);
      }

      return closest;
    },

    on: function (name, _) {
      return arguments.length > 1
        ? (event.on(name, _), simulation)
        : event.on(name);
    },

    // GPU-specific methods
    isGPUEnabled: function () {
      return gpuInitialized;
    },

    gpuReady: function () {
      return gpuReadyPromise;
    },

    async initGPU() {
      const success = await initGPU();
      if (success) createBuffers();
      return success;
    },
  });

  function configureGPUForce(name, force) {
    if (name === "charge" || name === "manyBody") {
      manyBodyConfig.enabled = true;
      // Extract settings from force if available
      if (force.strength) {
        const s = force.strength();
        manyBodyConfig.strength = typeof s === "function" ? -30 : s;
      }
      if (force.theta) manyBodyConfig.theta = force.theta();
      if (force.distanceMin) manyBodyConfig.distanceMin = force.distanceMin();
      if (force.distanceMax) manyBodyConfig.distanceMax = force.distanceMax();
    } else if (name === "center") {
      centerConfig.enabled = true;
      if (force.x) centerConfig.x = force.x();
      if (force.y) centerConfig.y = force.y();
      if (force.strength) centerConfig.strength = force.strength();
    } else if (name === "collide") {
      collisionConfig.enabled = true;
      if (force.radius) {
        const r = force.radius();
        collisionConfig.radius = typeof r === "function" ? 5 : r;
      }
      if (force.strength) collisionConfig.strength = force.strength();
      if (force.iterations)
        collisionConfig.iterations = force.iterations();
    } else if (name === "link") {
      // Extract link data
      if (force.links) {
        const links = force.links();
        linkData.links = links;
        linkData.distances = [];
        linkData.strengths = [];
        linkData.bias = [];

        // Get distances and strengths
        const distFn = force.distance ? force.distance() : () => 30;
        const strFn = force.strength ? force.strength() : null;

        // Count connections per node for bias calculation
        const count = new Array(nodes.length).fill(0);
        for (const link of links) {
          count[link.source.index] = (count[link.source.index] || 0) + 1;
          count[link.target.index] = (count[link.target.index] || 0) + 1;
        }

        for (let i = 0; i < links.length; i++) {
          const link = links[i];
          linkData.distances.push(
            typeof distFn === "function" ? distFn(link, i, links) : distFn
          );

          if (strFn) {
            linkData.strengths.push(
              typeof strFn === "function" ? strFn(link, i, links) : strFn
            );
          } else {
            // Default strength: 1 / min(count[source], count[target])
            linkData.strengths.push(
              1 /
                Math.min(
                  count[link.source.index],
                  count[link.target.index]
                )
            );
          }

          // Bias: count[source] / (count[source] + count[target])
          linkData.bias.push(
            count[link.source.index] /
              (count[link.source.index] + count[link.target.index])
          );
        }

        // Create link buffers
        if (gpuInitialized && links.length > 0) {
          if (linkBuffers) linkBuffers.destroy();
          linkBuffers = new LinkBuffers(gpuDevice, links.length);
          createBindGroups();
        }
      }
    } else if (name === "x") {
      forceXConfig.enabled = true;
      if (force.x) {
        const xVal = force.x();
        forceXConfig.x = typeof xVal === "function" ? 0 : xVal;
      }
      if (force.strength) {
        const s = force.strength();
        forceXConfig.strength = typeof s === "function" ? 0.1 : s;
      }
    } else if (name === "y") {
      forceYConfig.enabled = true;
      if (force.y) {
        const yVal = force.y();
        forceYConfig.y = typeof yVal === "function" ? 0 : yVal;
      }
      if (force.strength) {
        const s = force.strength();
        forceYConfig.strength = typeof s === "function" ? 0.1 : s;
      }
    } else if (name === "radial") {
      radialConfig.enabled = true;
      if (force.x) radialConfig.x = force.x();
      if (force.y) radialConfig.y = force.y();
      if (force.radius) {
        const r = force.radius();
        radialConfig.radius = typeof r === "function" ? 100 : r;
      }
      if (force.strength) {
        const s = force.strength();
        radialConfig.strength = typeof s === "function" ? 0.1 : s;
      }
    }
  }
}
