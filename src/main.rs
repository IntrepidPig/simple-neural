use std::{path::{Path, PathBuf}, fs};

use anyhow::Context;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Options {
    epochs: u32,
    batch_size: u32,
    rate: f32,
    #[structopt(short = "s", long = "save")]
    save: Option<PathBuf>,
    #[structopt(short = "l", long = "load")]
    load: Option<PathBuf>,
}

fn main() {
    let options = Options::from_args();

    let images = read_images(concat!(env!("CARGO_MANIFEST_DIR"), "/train-images-idx3-ubyte"));
    let labels = read_labels(concat!(env!("CARGO_MANIFEST_DIR"), "/train-labels-idx1-ubyte"))
        .into_iter()
        .map(|x| {
            let mut arr = Array1::<f32>::zeros(10);
            arr[x as usize] = 1.0;
            arr
        })
        .collect::<Vec<_>>();

    let test_images = read_images(concat!(env!("CARGO_MANIFEST_DIR"), "/t10k-images-idx3-ubyte"));
    let test_labels = read_labels(concat!(env!("CARGO_MANIFEST_DIR"), "/t10k-labels-idx1-ubyte"))
        .into_iter()
        .map(|x| {
            let mut arr = Array1::<f32>::zeros(10);
            arr[x as usize] = 1.0;
            arr
        })
        .collect::<Vec<_>>();

    let mut net = if let Some(ref load) = options.load {
        Net::load(load.as_ref()).expect("Failed to load network")
    } else {
        Net::new(vec![28 * 28, 15, 10])
    };

    for epoch in 0..options.epochs {
        net.train_batch(&images, &labels, options.batch_size as usize, options.rate);
        let accuracy = net.test(&test_images[..100], &test_labels[..100], convert_result);
        println!("{epoch}: {accuracy}");
    }

    let trained = net.test(&test_images, &test_labels, convert_result);
    println!("{trained}");

    if let Some(ref save) = options.save {
        net.save(save.as_ref());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Net {
    dims: Vec<usize>,
    /// weights and biases for each layer
    /// each W is an m x n matrix, m is number of neurons in next layer and n is number of neurons in
    /// previous layer.
    /// each B is a size m column vector, m is number of neurons in next layer
    /// column vector of values for each neuron in next layer given by o(WI+B), where I is column vector of inputs,
    /// and o(x) is elementwise sigma function. z = WI + B is the weighted input for an entire layer, z_i = W_ix I + B_i
    /// is the weighted input for the ith neuron of a layer (it is a row vector times column vector = scalar)
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
    // some state that is necessary for training
    z: Vec<Array1<f32>>,
    errors: Vec<Array1<f32>>,
    // rng for deterministic runs
    rng: rand_chacha::ChaCha12Rng,
}

impl Net {
    pub fn new(dims: Vec<usize>) -> Self {
        let mut this = Self {
            weights: (0..dims.len() - 1).map(|i| Array2::zeros([dims[i + 1], dims[i]])).collect(),
            biases: dims[1..].iter().map(|d| Array1::zeros(*d)).collect(),
            z: dims.iter().map(|d| Array1::zeros(*d)).collect(),
            errors: dims.iter().map(|d| Array1::zeros(*d)).collect(),
            dims,
            rng: rand_chacha::ChaCha12Rng::seed_from_u64(12),
        };
        this.randomize();
        this
    }

    pub fn save(&self, path: &Path) {
        let mut file = fs::File::create(path).expect("Failed to create save file");
        serde_json::to_writer_pretty(&mut file, self).expect("Failed to serialize network");
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut file = fs::File::open(path).context("Failed to open save file")?;
        serde_json::from_reader(&mut file).context("Failed to deserialize network")
    }

    pub fn randomize(&mut self) {
        for l in 0..self.weights.len() {
            for j in 0..self.weights[l].dim().0 {
                for k in 0..self.weights[l].dim().1 {
                    self.weights[l][[j, k]] = self.rng.gen::<f32>() * 2.0 - 1.0;
                }
                self.biases[l][j] = self.rng.gen::<f32>() * 2.0 - 1.0;
            }
        }
    }

    pub fn train_batch(&mut self, inputs: &[Array1<f32>], outputs: &[Array1<f32>], batch_size: usize, rate: f32) {
        let ls = self.dims.len(); // number of layers

        let mut dw_sums = self.weights.iter().map(|w| Array2::<f32>::zeros(w.dim())).collect::<Vec<_>>();
        let mut db_sums = self.biases.iter().map(|b| Array1::<f32>::zeros(b.dim())).collect::<Vec<_>>();

        for x in rand::seq::index::sample(&mut self.rng, inputs.len(), batch_size) {
            //println!("  calculating error from input {x}");
            let a = self.feedforward(inputs[x].clone());
            //dbg!(&a, &outputs[x]);
            //dbg!(&self.z);
            let y = outputs[x].clone();
            self.errors[ls - 1] = (a - y) * self.z[ls - 1].map(|z| dsigmoid(*z));
            for l in (0..ls - 1).rev() {
                self.errors[l] = self.weights[l].t().dot(&self.errors[l + 1]) * self.z[l].map(|z| dsigmoid(*z))
            }
            //dbg!(&self.errors);

            for l in 0..ls - 1 {
                db_sums[l] += &self.errors[l + 1];
                dw_sums[l] += &self.errors[l + 1]
                    .to_shape((self.dims[l + 1], 1))
                    .unwrap()
                    .dot(&self.z[l].map(|z| sigmoid(*z)).to_shape((1, self.dims[l])).unwrap());
            }

            for l in 0..ls - 1 {
                self.weights[l] = &self.weights[l] + &dw_sums[l] * (-rate / batch_size as f32);
                self.biases[l] = &self.biases[l] + &db_sums[l] * (-rate / batch_size as f32);
            }
        }
    }

    pub fn feedforward(&mut self, input: Array1<f32>) -> Array1<f32> {
        /* (0..self.weights.len()).fold(input, |v, i| {
            (self.weights[i].dot(&v) + &self.biases[i]).map(|z| sigmoid(*z))
        }) */
        self.z[0] = input.clone();
        let mut state = input;
        for l in 0..self.weights.len() {
            self.z[l + 1] = self.weights[l].dot(&state) + &self.biases[l];
            state = self.z[l + 1].map(|z| sigmoid(*z));
        }
        state
    }

    pub fn test<T: PartialEq, F: Fn(&Array1<f32>) -> T>(&mut self, inputs: &[Array1<f32>], outputs: &[Array1<f32>], convert: F) -> f32 {
        assert_eq!(inputs.len(), outputs.len());
        let mut success = 0;
        for i in 0..inputs.len() {
            let a = convert(&self.feedforward(inputs[i].clone()));
            let y = convert(&outputs[i]);
            if a == y {
                success += 1;
            }
        }
        success as f32 / inputs.len() as f32
    }
}

fn convert_result(output: &Array1<f32>) -> i32 {
    output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0 as i32
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn dsigmoid(z: f32) -> f32 {
    (-z).exp() / (1.0 + (-z).exp()).powi(2)
}

fn read_images(path: &str) -> Vec<Array1<f32>> {
    let data = std::fs::read(path).expect("Failed to open file");
    assert!(data.len() >= 16);
    let magic_number = u32::from_be_bytes(data[0..4].try_into().unwrap());
    let count = u32::from_be_bytes(data[4..8].try_into().unwrap());
    let width = u32::from_be_bytes(data[8..12].try_into().unwrap());
    let height = u32::from_be_bytes(data[12..16].try_into().unwrap());
    assert_eq!(magic_number, 0x00000803);
    assert_eq!(data.len() as u32, 16 + count * width * height);
    let size = (width * height) as usize;
    (0..(count as usize))
        .map(|i| Array1::from_iter(data[i * size..(i + 1) * size].iter().map(|v| *v as f32 / 255.0)))
        .collect()
}

fn read_labels(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).expect("Failed to open file");
    assert!(data.len() >= 8);
    let magic_number = u32::from_be_bytes(data[0..4].try_into().unwrap());
    let count = u32::from_be_bytes(data[4..8].try_into().unwrap());
    assert_eq!(magic_number, 0x00000801);
    assert_eq!(data.len() as u32, 8 + count);
    data[8..].to_owned()
}
