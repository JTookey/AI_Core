/// Layers module provides all the core functionality for creating and computing the layers of a Neural Network

use std::fmt::{self, Display};

use ndarray::{Zip, linalg};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::err::AIError;
use crate::util::*;
use crate::{AIVec, AIWeights};

/// Enumeration for the activation functions supported
/// - Sigmoid
/// - Tanh
/// - Rectified Linear
///
#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    Tanh,
    RectifiedLinear,
}

/// The Layer trait
/// 
pub trait Layer
{
    fn get_n_inputs(&self) -> usize;
    fn get_n_outputs(&self) -> usize;
    fn get_activation(&self) -> Activation;
    fn get_weight(&self, input_index: usize, output_index: usize) -> Option<&f64>;
    fn feedforward(&mut self, input: &AIVec, output: &mut AIVec);
    fn backproporgate(&mut self, input: &AIVec, error: &AIVec, backprop_error: &mut AIVec, learn_rate: f64);
}

/// The Layer Structure
/// 
pub struct BaseLayer
{
    n_inputs: usize,
    n_outputs: usize,
    input_weights: AIWeights,
    bias_weights: AIVec,
    activation_inputs: AIVec,
    activation_derivative: AIVec,
    delta: AIVec,
    activation_function: Activation,
    backprop_errors: AIVec,
}

impl BaseLayer {
    pub fn new( n_inputs: usize, n_outputs: usize, activation_function: Activation) -> Self {
        // Create weights between 1 and -1
        let input_weights = AIWeights::random((n_outputs, n_inputs), Uniform::new( -1.0, 1.0 ));
        // Create bias array initially 0
        let bias_weights = AIVec::zeros( n_outputs );
        // Create the required arrays
        let activation_inputs = AIVec::zeros( n_outputs );
        let output = AIVec::zeros( n_outputs );
        let error = AIVec::zeros( n_outputs );
        let total_error: f64 = 0.0;
        let activation_derivative = AIVec::zeros( n_outputs );
        let delta = AIVec::zeros( n_outputs ); // b_der
        //let w_der = AIWeights::zeros( (n_outputs, n_inputs) );
        let backprop_errors = AIVec::zeros( n_inputs );

        BaseLayer{
            n_inputs,
            n_outputs,
            input_weights,
            bias_weights,
            activation_inputs,
            activation_derivative,
            delta,
            activation_function,
            backprop_errors,
        }
    }
}

/// Implement the Layer trait
impl Layer for BaseLayer
{
    // Function to return the number of inputs to the layer
    fn get_n_inputs(&self) -> usize {
        self.n_inputs
    }

    // Function to return the number of outputs to the layer
    fn get_n_outputs(&self) -> usize {
        self.n_outputs
    }

    // Function to return the type of activation function the layer is using
    fn get_activation(&self) -> Activation {
        self.activation_function.clone()
    }

    fn get_weight(&self, input_index: usize, output_index: usize) -> Option<&f64> {
        self.input_weights.get([output_index,input_index])
    }

    fn feedforward(&mut self, input: &AIVec, output: &mut AIVec){
        // Step 1 - Calculate the Activation Input
        Zip::from(&mut self.activation_inputs).for_each(|x| *x = 0.0);
        linalg::general_mat_vec_mul(1.0, &self.input_weights, input, 1.0, &mut self.activation_inputs);
        self.activation_inputs += &self.bias_weights;

        // Step 2 - Calculate the Activation (i.e. the output) for Sigmoid
        Zip::from( output )
            .and( &self.activation_inputs )
            .for_each( | output, &input |  *output = sigmoid(input) );
    }

    fn  backproporgate(&mut self, input: &AIVec, error: &AIVec, backprop_error: &mut AIVec, learn_rate: f64) {
        // Step 1 - Calculate the Activation Input
        self.activation_inputs.fill(0.0);
        linalg::general_mat_vec_mul(1.0, &self.input_weights, &input.t(), 1.0, &mut self.activation_inputs);
        self.activation_inputs += &self.bias_weights;

        // Step 2 - Calculate the Activation Derivative (length of activation input)
        Zip::from( &mut self.activation_derivative)
            .and( &self.activation_inputs )
            .for_each( | a_d, &a_i | *a_d = sigmoid_derivative( a_i ) );

        // Step 3 - Calculate the Bias Derivative
        Zip::from( &mut self.delta )
            .and( &self.activation_derivative )
            .and( error )
            .for_each( |b_der, &act_der, &error| *b_der = act_der*error );

        
        // Step 4 - Calculate the Weight Derivative - (reusing b_der as this is act_d * error)
        for (i, delta) in self.delta.iter().enumerate() {
            for (j, inp) in input.iter().enumerate() {
                self.input_weights[[i,j]] -= delta * inp * learn_rate;
            }
        }

        // Step 5 - Update Weights
        self.bias_weights.zip_mut_with( &self.delta, | b, &d | *b -= d * learn_rate );

        // Step 6 - Calculate error to backproporgate
        backprop_error.fill(0.0);
        for input_index in 0..self.n_inputs {
            for output_index in 0..self.n_outputs {

                backprop_error[input_index] += &error[output_index] * &self.activation_derivative[output_index] * &self.input_weights[[output_index, input_index]];

            }
        }
    }
}


/// The Layer structure
pub struct BasicLayer {
    pub n_inputs: usize,
    pub n_outputs: usize,
    n_weights: usize,
    weights: Vec<f32>,
    pub activation_function: Activation,
    pub activation_inputs: Vec<f32>,
    pub output: Option<Vec<f32>>,
}

impl BasicLayer {
    pub fn new(n_inputs: usize, n_outputs: usize, activation_function: Activation) -> Self {

        let n_weights = n_inputs * n_outputs;

        let mut weights = Vec::with_capacity( n_weights );
        weights.resize( n_weights , 0.0 );

        let activation_inputs = Vec::with_capacity( n_inputs );
        
        BasicLayer {
            n_inputs,
            n_outputs,
            n_weights,
            weights,
            activation_function,
            activation_inputs,
            output: None
        }
    }

    pub fn new_with_rand(n_inputs: usize, n_outputs: usize, activation_function: Activation) -> Self {
        let mut new_layer = BasicLayer::new( n_inputs, n_outputs, activation_function);
        
        let rand_vec = new_random_vec::<f32>( new_layer.weights.len() );

        for i in 0..new_layer.weights.len() {
            new_layer.weights[i] = rand_vec[i];
        }

        new_layer
    }

    pub fn get_weight( &self, input_index: usize, output_index: usize) -> Result<f32, &'static str> {
        if input_index > self.n_inputs || output_index > self.n_outputs {
            return Err("Out of bounds");
        }

        Ok(self.weights[input_index * self.n_outputs + output_index])
    }

    pub fn process(&mut self, input: &Vec<f32>) -> Result<(), AIError> {
        if input.len() != self.n_inputs {
            println!("My n_inputs: {}, Vector given: {}", self.n_inputs, input.len());
            return Err(AIError::LengthMismatch);
        }

        self.gen_activation_inputs(input)?;
        self.output = match self.activation_function {
                    Activation::Sigmoid => Some(activation_sigmoid( &self.activation_inputs )),
                    Activation::Tanh => Some(activation_tanh( &self.activation_inputs )),
                    Activation::RectifiedLinear => Some(activation_rectified_linear( &self.activation_inputs)),
                };
        Ok(())
    }

    pub fn gen_activation_inputs(&mut self, input: &Vec<f32>) -> Result<Vec<f32>, AIError> {
        let mut output: Vec<f32> = Vec::with_capacity( self.n_outputs );

        if input.len() != self.n_inputs {
            return Err(AIError::InputMismatch);
        }

        for output_index in 0..self.n_outputs {

            let mut out_val = 0.0;

            for input_index in 0..self.n_inputs {
                out_val += input[input_index] * self.get_weight(input_index, output_index).unwrap();
            }

            output.push(out_val);
        }

        self.activation_inputs = output.clone();

        Ok(output)
    }

    pub fn calc_output_weight_derivatives( &self, input: &Vec<f32>, output_error: &Vec<f32>, activation_derivative: &Vec<f32>) -> Result<Vec<f32>, AIError> {
        if self.n_inputs == input.len() && self.n_outputs == output_error.len() && self.n_outputs == activation_derivative.len() {

            let mut weight_derivatives = self.weights.clone();

            for input_index in 0..self.n_inputs {
                for output_index in 0..self.n_outputs {
                    weight_derivatives[ input_index * self.n_outputs + output_index ] = input[input_index] * output_error[output_index] * activation_derivative[output_index];
                }
            }

            Ok(weight_derivatives)

        } else {
            Err(AIError::LengthMismatch)
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

impl Display for BasicLayer {
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

/// Sigmoid Function - single value
fn sigmoid(input: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * input).exp())
}

/// Sigmoid Derivative - single value
fn sigmoid_derivative(input: f64) -> f64 {
    let s = sigmoid(input);
    s * (1.0 - s)
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

pub fn calc_output_layer_error( output: &Vec<f32>, expected: &Vec<f32> ) -> Result<Vec<f32>, AIError> {
    if output.len() == expected.len() {
        let mut error = output.clone();

        for i in 0..error.len() {
            error[i] -= expected[i];
        }

        Ok( error )
    } else {
        Err(AIError::LengthMismatch)
    }
}

/// Function to calc the max of a vector
pub fn calc_vec_max( input: &Vec<f32> ) -> Vec<f32> {
    let mut max: f32 = std::f32::MIN;
    let mut max_idx = 0;
    let l = input.len();
    let mut output: Vec<f32> = Vec::with_capacity(l);
    for i in 0..l {
        output.push(0.0);

        if input[i] > max {
            max = input[i];
            max_idx = i;
        }
    }
    output[max_idx] = 1.0;

    output
}

/// Function for normalising a vector between a couple of values
// pub fn normalise( input: &Vec<f32>, input_min: f32, input_max: f32, output_min: f32, output_max: f32 ) -> Vec<f32> {
//     // Create the vector
//     let mut output: Vec<f32> = Vec::new();

//     for val in input {
//         output.push( lin_interp(*val, input_min, input_max, output_min, output_max) );
//     }

//     // return the output vector
//     output
// }

/// Function for element wise multiplication
pub fn multiply_vec( in_1: &Vec<f32>, in_2: &Vec<f32>) -> Result<Vec<f32>, AIError> {
    if in_1.len() != in_2.len() {
        return Err(AIError::LengthMismatch);
    }

    let mut out: Vec<f32> = Vec::new();
    
    for i in 0..in_1.len() {
        out.push( in_1[i] * in_2[i] );
    }

    Ok(out)
}