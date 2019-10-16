use std::fmt::{self, Display};
use rand::prelude::*;

pub struct Layer {
    n_inputs: usize,
    n_outputs: usize,
    n_weights: usize,
    weights: Vec<f32>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Layer {
        
        let n_weights = n_inputs * n_outputs;

        let mut weights = Vec::with_capacity( n_weights );
        weights.resize( n_weights , 0.0 );
        
        Layer {
            n_inputs,
            n_outputs,
            n_weights,
            weights,
        }
    }

    pub fn new_with_rand(n_inputs: usize, n_outputs: usize) -> Layer {
        let mut new_layer = Layer::new( n_inputs, n_outputs );
        
        let mut rng = rand::thread_rng();

        for i in 0..new_layer.weights.len() {
            new_layer.weights[i] = rng.gen();
        }

        new_layer
    }

    pub fn get_weight( &self, input_index: usize, output_index: usize) -> Result<f32, &'static str> {
        if input_index > self.n_inputs || output_index > self.n_outputs {
            return Err("Out of bounds");
        }

        Ok(self.weights[input_index * self.n_outputs + output_index])
    }

    pub fn gen_activation_inputs(&self, input: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
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

    pub fn calc_output_weight_derivatives( &self, input: &Vec<f32>, output_error: &Vec<f32>, activation_derivative: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if self.n_inputs == input.len() && self.n_outputs == output_error.len() && self.n_outputs == activation_derivative.len() {

            let mut weight_derivatives = self.weights.clone();

            for input_index in 0..self.n_inputs {
                for output_index in 0..self.n_outputs {
                    weight_derivatives[ input_index * self.n_outputs + output_index ] = input[input_index] * output_error[output_index] * activation_derivative[output_index];
                }
            }

            Ok(weight_derivatives)

        } else {
            Err("Vector length error")
        }
    }

    pub fn calc_backprop_errors( &self, out_error: Vec<f32>, activation_derivative: Vec<f32> ) -> Vec<f32> {
        let mut backprop_errors = Vec::with_capacity( self.n_inputs );
        backprop_errors.resize( self.n_inputs, 0.0 );

        for input_index in 0..self.n_inputs {
            for output_index in 0..self.n_outputs {

                backprop_errors[input_index] += out_error[output_index] * activation_derivative[output_index] * self.get_weight(input_index, output_index).unwrap();

            }
        }

        // return
        backprop_errors
    }

    pub fn update_weights( &mut self, learn_rate: f32, weight_derivatives: &Vec<f32> ) {

        if weight_derivatives.len() == self.n_weights {
            for index in 0..self.weights.len() {
                self.weights[ index ] -= learn_rate * weight_derivatives[ index ];
            }
        }

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

pub fn activation_sigmoid( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = 1.0 / ( 1.0 + (-1.0 * output[i]).exp() );
    }

    // return
    output
}

pub fn derivative_sigmoid( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        let s = 1.0 / ( 1.0 + (-1.0 * output[i]).exp() );
        output[i] = s * (1.0 - s);
    }

    // return
    output
}

pub fn activation_tanh( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = output[i].tanh();
    }

    // return
    output
}

pub fn derivative_tanh( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = ( 1.0 / output[i].cosh() ).powi(2);
    }

    // return
    output
}

pub fn activation_rectified_linear( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = output[i].max( 0.0 );
    }

    // return
    output
}

pub fn derivative_rectified_linear( input: &Vec<f32> ) -> Vec<f32> {
    let mut output = input.clone();

    for i in 0..output.len() {
        output[i] = if output[i] > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    // return
    output
}

pub fn calc_output_layer_error( output: &Vec<f32>, expected: &Vec<f32> ) -> Result<Vec<f32>, &'static str> {
    if output.len() == expected.len() {
        let mut error = output.clone();

        for i in 0..error.len() {
            error[i] -= expected[i];
        }

        Ok( error )
    } else {
        Err("Output Vector and Expected Vector need to be the same length")
    }
}

/// Function for normalising a vector between a couple of values
pub fn normalise( input: &Vec<f32>, input_min: f32, input_max: f32, output_min: f32, output_max: f32 ) -> Vec<f32> {
    // Create the vector
    let mut output: Vec<f32> = Vec::new();

    for val in input {
        output.push( lin_interp(*val, input_min, input_max, output_min, output_max) );
    }

    // return the output vector
    output
}

/// Function to linearly interpolate
pub fn lin_interp( input: f32, input_min: f32, input_max: f32, output_min: f32, output_max: f32 ) -> f32 {
    output_min + (output_max - output_min) * (input - input_min) / (input_max - input_min)
}

/// Function to calculate error
pub fn calc_average_sum_square( vector: &Vec<f32> ) -> f32 {
    let mut average_sum_square: f32 = 0.0;

    for index in 0..vector.len() {
        average_sum_square += vector[index].powi(2);
    }

    average_sum_square /= vector.len() as f32;

    // return
    average_sum_square
}