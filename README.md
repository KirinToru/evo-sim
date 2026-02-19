# Evolution Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2021-orange?style=flat-square&logo=rust">
  <img src="https://img.shields.io/badge/Platform-Wasm-blue?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
</p>

A simulation of evolution using neural networks and genetic algorithms, running in the browser using WebAssembly. Based on the [pwy.io series](https://pwy.io/posts/learning-to-fly-pt1/).

## About

This project simulates simple creatures ("birdies") evolving to survive and complete tasks. It implements:
- **Neural Networks**: For decision making (`libs/neural-network`).
- **Genetic Algorithms**: For evolution and improvement over generations (`libs/genetic-algorithm`).
- **Simulation Logic**: The core world and creature logic (`libs/simulation`).
- **Web App**: Visualization using Rust -> WebAssembly (`app`).

## Structure

- **`app/`**: The web application (Rust Wasm + HTML/JS).
- **`libs/`**: Core logic libraries.
    - `genetic-algorithm`: Generic GA implementation.
    - `neural-network`: Simple neural network implementation.
    - `simulation`: The game/world logic.

## Usage

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (latest stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

### Build and Run

1. **Build the Wasm module:**
   ```bash
   cd app
   wasm-pack build --target web
   ```

2. **Serve the application:**
   You need a local web server to serve the `app/www` directory.
   ```bash
   # If you have python installed:
   cd app/www
   python -m http.server
   ```
   Then open `http://localhost:8000` in your browser.

## Release

To create a release, you typically:
1. Update version numbers in `Cargo.toml`.
2. Commit changes.
3. Tag the release.
4. Push the tag.

## License

MIT License. See [LICENSE](LICENSE) for details.
