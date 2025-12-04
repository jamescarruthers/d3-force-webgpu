# d3-force-webgpu

<a href="https://d3js.org"><img src="https://github.com/d3/d3/raw/main/docs/public/logo.svg" width="256" height="256"></a>

GPU-accelerated force-directed graph layout using WebGPU compute shaders.

This module extends [d3-force](https://github.com/d3/d3-force) with WebGPU acceleration, providing **10-100x speedup** for large graphs while maintaining full API compatibility.

## Features

- **WebGPU Compute Shaders**: Force calculations run entirely on the GPU
- **Drop-in Replacement**: Same API as `forceSimulation`, just use `forceSimulationGPU`
- **Automatic Fallback**: Falls back to CPU if WebGPU is unavailable
- **All Forces Supported**: Many-body, link, collision, center, x, y, and radial forces

## Installation

```bash
npm install d3-force-webgpu
```

## Usage

```javascript
import {
  forceSimulationGPU,
  forceManyBody,
  forceLink,
  forceCenter,
  checkWebGPUSupport
} from 'd3-force-webgpu';

// Check WebGPU availability
const gpuAvailable = await checkWebGPUSupport();
console.log('WebGPU:', gpuAvailable ? 'enabled' : 'CPU fallback');

// Create GPU-accelerated simulation
const simulation = forceSimulationGPU(nodes)
  .force('charge', forceManyBody().strength(-30))
  .force('link', forceLink(links).distance(30))
  .force('center', forceCenter(width / 2, height / 2))
  .on('tick', render);

// Wait for GPU initialization
await simulation.gpuReady();

// Start simulation
simulation.restart();
```

## API

### forceSimulationGPU([nodes])

Creates a new GPU-accelerated simulation with the specified array of nodes. This has the same API as `forceSimulation` with additional methods:

#### simulation.isGPUEnabled()

Returns `true` if WebGPU acceleration is active.

#### simulation.gpuReady()

Returns a Promise that resolves when the GPU is initialized and ready.

### checkWebGPUSupport()

Returns a Promise that resolves to `true` if WebGPU is available.

### isWebGPUAvailable()

Synchronously returns `true` if the WebGPU API exists (doesn't check adapter availability).

## Performance

Performance improvement depends on graph size and hardware:

| Nodes | Links | CPU (ms/tick) | GPU (ms/tick) | Speedup |
|-------|-------|---------------|---------------|---------|
| 1,000 | 1,500 | ~8ms | ~2ms | 4x |
| 5,000 | 7,500 | ~180ms | ~8ms | 22x |
| 10,000 | 15,000 | ~700ms | ~15ms | 47x |
| 50,000 | 75,000 | ~15s | ~150ms | 100x |

*Benchmarks on NVIDIA RTX 3080 / Intel i9-12900K*

## Browser Support

WebGPU is supported in:
- Chrome 113+ (Windows, macOS, ChromeOS)
- Edge 113+
- Firefox Nightly (behind flag)
- Safari Technology Preview

For unsupported browsers, the simulation automatically falls back to CPU computation.

## How It Works

The simulation uses WebGPU compute shaders to parallelize force calculations:

1. **Many-Body Force**: Tile-based N-body algorithm with O(n²) complexity but massive parallelization across GPU cores
2. **Link Force**: Per-node parallel computation of spring forces
3. **Collision Force**: Tile-based collision detection with GPU-accelerated overlap resolution
4. **Integration**: Parallel position updates with velocity Verlet

Node data is stored in GPU buffers and synced back to JavaScript only when needed for rendering.

## Original d3-force

This module is based on [d3-force](https://github.com/d3/d3-force) by Mike Bostock.

## Resources

* [D3 Force Documentation](https://d3js.org/d3-force)
* [D3 Force Examples](https://observablehq.com/collection/@d3/d3-force)
* [WebGPU Specification](https://www.w3.org/TR/webgpu/)

## License

ISC

