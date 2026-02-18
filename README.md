# Evolution Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2024-orange?style=flat-square&logo=rust">
  <img src="https://img.shields.io/badge/Platform-Windows-blue?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
</p>

A simulation of evolution using neural networks and genetic algorithms, based on the [pwy.io series](https://pwy.io/posts/learning-to-fly-pt1/).

## About

This project simulates simple creatures ("birdies") evolving to survive and complete tasks. It implements:
- **Neural Networks**: For decision making.
- **Genetic Algorithms**: For evolution and improvement over generations.

Currently, it includes the core library logic (`lib-neural-network`) and a basic app structure.

## Structure

- **`app/`**: The main application entry point.
- **`libs/neural-network/`**: The neural network and genetic algorithm library.

## Usage

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (latest stable)

### Build and Run

```bash
# Run the application
cargo run

# Run tests
cargo test
```

## License

MIT License. See [LICENSE](LICENSE) for details.
