use std::fmt::{self, Display};
use rand::prelude::*;

fn main() {
    println!("I think and therefore I am!\n");

    // Create simple net
    let input: Vec<f32> = vec![1.0, -1.0];

    // Normalise
    let norm_input = normalise( &input , 0.0 , 1.0, 0.0, 1.0);
    println!("Input Vector: {:.3?}\n",norm_input);


    // Create a layer
    let new_layer = Layer::new_with_rand( 2, 2 );
    println!("Weights:\n{}\n", new_layer);


    // Generate Output
    let output = new_layer.gen_output( &norm_input ).unwrap();
    println!("Output Vector: {:.3?}\n",output);
}

struct Layer {
    n_inputs: usize,
    n_outputs: usize,
    weights: Vec<f32>,
}

impl Layer {
    fn new(n_inputs: usize, n_outputs: usize) -> Layer {
        
        let mut weights = Vec::with_capacity( n_inputs * n_outputs );
        weights.resize( n_inputs * n_outputs , 0.0 );
        
        Layer {
            n_inputs,
            n_outputs,
            weights,
        }
    }

    fn new_with_rand(n_inputs: usize, n_outputs: usize) -> Layer {
        let mut new_layer = Layer::new( n_inputs, n_outputs );
        
        let mut rng = rand::thread_rng();

        for i in 0..new_layer.weights.len() {
            new_layer.weights[i] = rng.gen();
        }

        new_layer
    }

    fn get_weight( &self, input_index: usize, output_index: usize) -> Result<f32, &'static str> {
        if ( input_index > self.n_inputs || output_index > self.n_outputs ) {
            return Err("Out of bounds");
        }

        Ok(self.weights[input_index * self.n_outputs + output_index])
    }

    fn gen_output(&self, input: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
        let mut output: Vec<f32> = Vec::with_capacity( self.n_outputs );

        if input.len() != self.n_inputs {
            return Err("Input to layer is the wrong size");
        }

        for output_index in 0..self.n_outputs {

            let mut out_val = 0.0;

            for input_index in 0..self.n_inputs {
                out_val += input[input_index] * self.get_weight(input_index, output_index).unwrap();
            }

            output.push(out_val);
        }

        Ok(output)
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.

        for input_index in 0..self.n_inputs {
            for output_index in 0..self.n_outputs {
                write!(f, "{:.3}, ", self.weights.get(input_index * self.n_outputs + output_index ).unwrap_or(&9.999) )?;
            }
            
            if input_index < self.n_inputs - 1 {
                writeln!(f,"")?;
            }
        }
        write!(f,"")
    }
}

fn activation_sigmoid( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = 1.0 / ( 1.0 + (-1.0 * output[i]).exp() );
    }

    // return
    output
}


fn activation_tanh( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = output[i].tanh();
    }

    // return
    output
}

fn activation_rectified_linear( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = output[i].max( 0.0 );
    }

    // return
    output
}

/// Function for normalising a vector between a couple of values
fn normalise( input: &Vec<f32>, input_min: f32, input_max: f32, output_min: f32, output_max: f32 ) -> Vec<f32> {
    // Create the vector
    let mut output: Vec<f32> = Vec::new();

    for val in input {
        output.push( lin_interp(*val, input_min, input_max, output_min, output_max) );
    }

    // return the output vector
    output
}

/// Function to linearly interpolate
fn lin_interp( input: f32, input_min: f32, input_max: f32, output_min: f32, output_max: f32 ) -> f32 {
    output_min + (output_max - output_min) * (input - input_min) / (input_max - input_min)
}