# d3-force-webgpu

GPU-accelerated force-directed graph layout with intelligent CPU/GPU selection. A drop-in replacement for [d3-force](https://github.com/d3/d3-force) that automatically chooses between CPU and WebGPU implementations for optimal performance.

[![npm version](https://img.shields.io/npm/v/d3-force-webgpu.svg)](https://www.npmjs.com/package/d3-force-webgpu)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](https://opensource.org/licenses/ISC)

## Features

- 🧠 **Adaptive Mode**: Automatically chooses CPU or GPU based on graph complexity
- 🚀 **WebGPU Acceleration**: Up to 10x performance improvement for large graphs (1000+ nodes)
- 💻 **CPU Fallback**: Seamless fallback to CPU for smaller graphs and unsupported browsers
- 🔧 **Drop-in Replacement**: Compatible with existing d3-force code
- 📊 **Performance Monitoring**: Real-time FPS tracking with dynamic mode switching
- 🎛️ **Manual Override**: Force CPU or GPU mode when needed

## Installation

```bash
npm install d3-force-webgpu
```

## Quick Start

### Adaptive Mode (Recommended)

```javascript
import * as d3 from 'd3-force-webgpu';

// Create nodes and links
const nodes = [{id: 1}, {id: 2}, {id: 3}];
const links = [{source: 1, target: 2}, {source: 2, target: 3}];

// Adaptive simulation automatically chooses CPU or GPU
const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id))
  .force('charge', d3.forceManyBody())
  .force('center', d3.forceCenter(width / 2, height / 2));
```

### Manual Mode Selection

```javascript
// Force CPU mode
import { forceCPU } from 'd3-force-webgpu';
const cpuSimulation = forceCPU(nodes);

// Force GPU mode  
import { forceGPU } from 'd3-force-webgpu';
const gpuSimulation = forceGPU(nodes);

// Specific imports
import { forceSimulationCPU, forceSimulationGPU } from 'd3-force-webgpu/cpu';
```

## Performance Guidelines

The adaptive system automatically chooses the optimal implementation:

| Graph Size | Recommended Mode | Performance Gain |
|------------|------------------|------------------|
| < 200 nodes | CPU | Baseline |
| 200-750 nodes | CPU | CPU overhead < GPU benefit |
| 750-2000 nodes | Adaptive | Varies by complexity |
| 2000+ nodes | GPU | 2-10x improvement |

## API Reference

### forceSimulation(nodes, options?)

Creates an adaptive simulation that automatically chooses between CPU and GPU.

**Options:**
- `mode`: `'auto'` (default), `'cpu'`, or `'gpu'`
- `enableSwitching`: `boolean` - Allow dynamic switching (default: `true`)
- `onModeChange`: `function` - Callback when mode changes
- `estimatedLinkCount`: `number` - Help determine complexity

```javascript
const simulation = d3.forceSimulation(nodes, {
  mode: 'auto',
  enableSwitching: true,
  onModeChange: (mode, reason) => {
    console.log(`Switched to ${mode}: ${reason}`);
  }
});
```

### Adaptive Methods

```javascript
// Get current mode
const mode = simulation.getCurrentMode(); // 'cpu' or 'gpu'

// Get performance stats
const stats = simulation.getPerformanceStats();
console.log(`FPS: ${stats.avgFPS}, Mode: ${stats.mode}`);

// Force mode change
await simulation.forceMode('gpu');
```

### Utility Functions

```javascript
import { getRecommendedMode, createOptimalSimulation } from 'd3-force-webgpu';

// Get recommendation without creating simulation
const rec = getRecommendedMode(nodes, links);
console.log(`Recommended: ${rec.mode} - ${rec.reason}`);

// Create simulation with optimal settings
const simulation = createOptimalSimulation(nodes, {
  links: links,
  onModeChange: (mode, reason) => console.log(mode, reason)
});
```

## Browser Support

- **WebGPU**: Chrome 113+, Edge 113+, Opera 99+
- **CPU Fallback**: All modern browsers
- **Automatic Detection**: Falls back to CPU if WebGPU unavailable

## Examples

### Basic Usage

```javascript
import * as d3 from 'd3-force-webgpu';

const nodes = d3.range(1000).map(i => ({id: i}));
const links = d3.range(999).map(i => ({source: i, target: i + 1}));

const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id))
  .force('charge', d3.forceManyBody().strength(-300))
  .force('center', d3.forceCenter(400, 300));

simulation.on('tick', () => {
  // Update visualization
  updateVisualization(nodes, links);
});
```

### Performance Monitoring

```javascript
const simulation = d3.forceSimulation(nodes, {
  onModeChange: (mode, reason) => {
    console.log(`🔄 ${mode.toUpperCase()}: ${reason}`);
  }
});

// Monitor performance
setInterval(() => {
  const stats = simulation.getPerformanceStats();
  console.log(`FPS: ${stats.avgFPS}, Frame: ${stats.avgFrameTime}ms`);
}, 1000);
```

### Large Graph Example

```javascript
// For large graphs, the system will automatically use GPU
const largeNodes = d3.range(5000).map(i => ({
  id: i,
  group: Math.floor(i / 100)
}));

const largeLinks = [];
for (let i = 0; i < 10000; i++) {
  largeLinks.push({
    source: Math.floor(Math.random() * 5000),
    target: Math.floor(Math.random() * 5000)
  });
}

const simulation = d3.forceSimulation(largeNodes)
  .force('link', d3.forceLink(largeLinks).id(d => d.id))
  .force('charge', d3.forceManyBody().strength(-30))
  .force('center', d3.forceCenter(800, 600));

// Will automatically use GPU for optimal performance
```

## Migration from d3-force

This library is a drop-in replacement for d3-force:

```javascript
// Before
import * as d3 from 'd3-force';

// After  
import * as d3 from 'd3-force-webgpu';

// All existing code works unchanged!
const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links))
  .force('charge', d3.forceManyBody())
  .force('center', d3.forceCenter());
```

## Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Build for production
npm run build
```

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

ISC License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on top of the excellent [d3-force](https://github.com/d3/d3-force) library by Mike Bostock and contributors.

