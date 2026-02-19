use lib_genetic_algorithm as ga;
use lib_neural_network as nn;
use rand::prelude::*;
use serde::Serialize;

pub struct Simulation {
    world: World,
    ga: ga::GeneticAlgorithm<ga::RouletteWheelSelection>,
    age: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct World {
    pub animals: Vec<Animal>,
    pub foods: Vec<Food>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Food {
    pub position: Position,
}

#[derive(Clone, Debug, Serialize)]
pub struct Animal {
    pub position: Position,
    pub rotation: Rotation,
    pub speed: f32,
    #[serde(skip)]
    pub brain: Brain,
    #[serde(skip)]
    pub chromosome: ga::Chromosome,
    /// Number of foods eaten
    pub satiation: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Rotation {
    pub angle: f32,
}

pub type Brain = nn::Network;

impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let world = World::random(rng);
        let ga = ga::GeneticAlgorithm::new(
            ga::RouletteWheelSelection,
            ga::UniformCrossover,
            ga::GaussianMutation::new(0.01, 0.3),
        );

        Self { world, ga, age: 0 }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn train(&mut self, rng: &mut dyn RngCore) -> ga::Statistics {
        loop {
            if let Some(summary) = self.step(rng) {
                return summary;
            }
        }
    }

    pub fn step(&mut self, rng: &mut dyn RngCore) -> Option<ga::Statistics> {
        self.process_collisions(rng);
        self.process_brains();
        self.process_movements();

        self.age += 1;

        if self.age > 2500 {
            Some(self.evolve(rng))
        } else {
            None
        }
    }

    fn process_collisions(&mut self, rng: &mut dyn RngCore) {
        for animal in &mut self.world.animals {
            for food in &mut self.world.foods {
                let distance = animal.position.distance(&food.position);

                if distance <= 0.01 {
                    animal.satiation += 1;
                    food.position = rng.random_position();
                }
            }
        }
    }

    fn process_brains(&mut self) {
        for animal in &mut self.world.animals {
            let vision = animal.eye_matrix(&self.world.foods);

            let response = animal.brain.propagate(vision);

            let speed = response[0].clamp(0.0, 1.0);
            let rotation = response[1].clamp(0.0, 1.0);

            animal.speed = (animal.speed + speed * 0.00002).clamp(0.0002, 0.001);
            animal.rotation.angle += (rotation - 0.5) * 0.1;
        }
    }

    fn process_movements(&mut self) {
        for animal in &mut self.world.animals {
            animal.position.x += animal.rotation.angle.cos() * animal.speed;
            animal.position.y += animal.rotation.angle.sin() * animal.speed;

            animal.position.x = animal.position.x.rem_euclid(1.0);
            animal.position.y = animal.position.y.rem_euclid(1.0);
        }
    }

    fn evolve(&mut self, rng: &mut dyn RngCore) -> ga::Statistics {
        self.age = 0;

        // Evolve current population
        // genetic-algorithm's evolve returns NEW population (created via Individual::create).
        // create() initializes them at default position (0.5, 0.5).
        // We probably want to randomize their positions here.

        let (mut new_animals, stats) = self.ga.evolve(rng, &self.world.animals);

        // Randomize positions for new generation
        for animal in &mut new_animals {
            animal.position = rng.random_position();
            animal.rotation = rng.random_rotation();
        }

        self.world.animals = new_animals;

        for food in &mut self.world.foods {
            food.position = rng.random_position();
        }

        stats
    }
}

impl World {
    fn random(rng: &mut dyn RngCore) -> Self {
        let animals = (0..40).map(|_| Animal::random(rng)).collect();
        let foods = (0..60).map(|_| Food::random(rng)).collect();

        Self { animals, foods }
    }
}

impl Animal {
    fn random(rng: &mut dyn RngCore) -> Self {
        let brain = Brain::random(rng, &Self::topology());
        let chromosome = brain.as_chromosome();

        Self {
            position: rng.random_position(),
            rotation: rng.random_rotation(),
            speed: 0.0002,
            brain,
            chromosome,
            satiation: 0,
        }
    }

    fn topology() -> [nn::LayerTopology; 3] {
        [
            nn::LayerTopology { neurons: 9 }, // Vision
            nn::LayerTopology { neurons: 16 },
            nn::LayerTopology { neurons: 2 }, // Speed, Rotation
        ]
    }

    fn eye_matrix(&self, _foods: &[Food]) -> Vec<f32> {
        // Placeholder for eye matrix.
        // Logic: for each "eye ray", return distance to nearest food or 0.
        // Assuming 9 inputs.
        vec![0.0; 9]
    }
}

impl ga::Individual for Animal {
    fn create(chromosome: ga::Chromosome) -> Self {
        let brain = Brain::from_chromosome(chromosome.clone(), &Self::topology());

        Self {
            position: Position { x: 0.5, y: 0.5 },
            rotation: Rotation { angle: 0.0 },
            speed: 0.0002,
            brain,
            chromosome,
            satiation: 0,
        }
    }

    fn fitness(&self) -> f32 {
        self.satiation as f32
    }

    fn chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }
}

impl Position {
    pub fn distance(&self, other: &Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

trait RandomPosition {
    fn random_position(&mut self) -> Position;
    fn random_rotation(&mut self) -> Rotation;
}

impl<R: Rng + ?Sized> RandomPosition for R {
    fn random_position(&mut self) -> Position {
        Position {
            x: self.gen(), // random() replaces gen() in rand 0.9+
            y: self.gen(),
        }
    }

    fn random_rotation(&mut self) -> Rotation {
        Rotation {
            angle: self.gen::<f32>() * 2.0 * std::f32::consts::PI,
        }
    }
}

impl Food {
    fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.random_position(),
        }
    }
}

trait BrainExt {
    fn from_chromosome(chromosome: ga::Chromosome, topology: &[nn::LayerTopology]) -> Self;
    fn as_chromosome(&self) -> ga::Chromosome;
}

impl BrainExt for Brain {
    fn from_chromosome(chromosome: ga::Chromosome, topology: &[nn::LayerTopology]) -> Self {
        Self::from_weights(topology, chromosome)
    }

    fn as_chromosome(&self) -> ga::Chromosome {
        self.weights().collect()
    }
}
