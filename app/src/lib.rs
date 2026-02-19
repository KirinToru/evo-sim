use lib_simulation::Simulation;
use rand::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SimulationWasm {
    sim: Simulation,
    rng: ThreadRng,
}

#[wasm_bindgen]
impl SimulationWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let sim = Simulation::random(&mut rng);

        Self { sim, rng }
    }

    pub fn world(&self) -> JsValue {
        let world = self.sim.world();
        serde_wasm_bindgen::to_value(world).unwrap()
    }

    pub fn step(&mut self) -> String {
        match self.sim.step(&mut self.rng) {
            Some(stats) => format!(
                "Min: {:.2}, Max: {:.2}, Avg: {:.2}",
                stats.min_fitness, stats.max_fitness, stats.avg_fitness
            ),
            None => String::new(),
        }
    }

    pub fn train(&mut self) -> String {
        let stats = self.sim.train(&mut self.rng);
        format!(
            "Min: {:.2}, Max: {:.2}, Avg: {:.2}",
            stats.min_fitness, stats.max_fitness, stats.avg_fitness
        )
    }
}
