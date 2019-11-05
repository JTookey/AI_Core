use std::error;
use std::fmt::{self, Display};
use rand::prelude::*;

// Define our error types. These may be customized for our error handling cases.
// Now we will be able to write our own errors, defer to an underlying error
// implementation, or do something in between.
#[derive(Debug, Clone)]
pub enum AIError{
    Unprocessed,
    InputMismatch,
    LengthMismatch,
}

// Generation of an error is completely separate from how it is displayed.
// There's no need to be concerned about cluttering complex logic with the display style.
//
// Note that we don't store any extra info about the errors. This means we can't state
// which string failed to parse without modifying our types to carry that information.
impl fmt::Display for AIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AIError::Unprocessed => {
                write!(f, "Network needs processing")
            },
            AIError::InputMismatch => { 
                write!(f, "Input does not match")
            },
            AIError::LengthMismatch => { 
                write!(f, "Wrong number of inputs provided to network")
            }
        }
    }
}

// This is important for other errors to wrap this one.
impl error::Error for AIError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    Tanh,
    RectifiedLinear,
}

pub struct NetworkBuilder{
    n_inputs: usize,
    n_outputs: usize,
    layers: Vec<(usize, usize, Activation)>,
}

impl NetworkBuilder {
    pub fn new(n_inputs: usize) -> NetworkBuilder {
        NetworkBuilder{
            n_inputs,
            n_outputs: 0,
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, n_nodes: usize, activation_function: Activation) -> &mut NetworkBuilder {
        if self.layers.len() == 0 {
            self.layers.push( (self.n_inputs, n_nodes, activation_function) );
        } else {
            let last_n_outputs = self.layers.last().unwrap().1;
            self.layers.push( (last_n_outputs, n_nodes, activation_function) );
        }
        
        self.n_outputs = n_nodes;

        self
    }

    pub fn build(&self) -> Option<NeuralNetwork> {
        let mut layers: Vec<Layer> = Vec::new();

        for (n_inputs, n_outputs, activation_function) in &self.layers {
            layers.push( Layer::new_with_rand(*n_inputs, *n_outputs, activation_function.clone()) );
        }

        // Create the NeuralNetwork struct
        Some(NeuralNetwork {
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            layers,
            last_input: None,
        })
    }
}

pub struct NeuralNetwork {
    n_inputs: usize,
    n_outputs: usize,
    layers: Vec<Layer>,
    last_input: Option<Vec<f32>>,
}

impl NeuralNetwork {
    fn check_input(&self, input: &Vec<f32>) -> Result<(), AIError> {

        // Check there was a last value
        if let Some(l_in) = &self.last_input {
            // Check the length
            if l_in.len() != input.len() {
                return Err(AIError::LengthMismatch);
            }
            // Check values
            for (i, in_value) in l_in.iter().enumerate() {
                if input[i] != *in_value {
                    return Err(AIError::InputMismatch);
                }
            }
        } else {
            return Err(AIError::Unprocessed);
        }

        Ok(())
    }

    pub fn feedforward(&mut self, input: &Vec<f32>) -> Result<Vec<f32>, AIError> {
        if let Ok(()) = self.check_input(input) {
            self.process()?;
        } else {
            self.last_input = Some(input.clone());
            self.process()?;
        }

        let last_layer = self.layers.last().unwrap();
        if let Some(output) = &last_layer.output {
            Ok( output.clone() )
        }  else {
            Err(AIError::InputMismatch)
        }
        
    }

    fn process(&mut self) -> Result<(), AIError> {
        let mut layer_input = self.last_input.clone();

        for layer in &mut self.layers{
            if let Some( input ) = layer_input {
                layer.process( &input )?;
                layer_input = layer.output.clone();
            }
        }

        Ok(())
    }

    pub fn backproporgate(&mut self, input: &Vec<f32>, result: &Vec<f32>) -> Result<(), AIError> {
        if let Err(_) = self.check_input(input){
            self.last_input = Some(input.clone());
            self.process()?;
        };

        // Calc errors specifically for the network output
        let last_layer = self.layers.last().unwrap();
        let mut backprop_errors = calc_output_layer_error( &last_layer.output.as_ref().unwrap(), &result )?;
        
        // Calcs error as a scalar value
        //let current_error = calc_average_sum_square(&error)
        for i in (0..self.layers.len()).rev() {
            let (layers_left, layers_right) = self.layers.split_at_mut(i);
            let layer = layers_right.first_mut().unwrap();   
            let activation_derivative = derivative_sigmoid(&layer.activation_inputs);
            
            let layer_input: &Vec<f32> = if i==0 {
                input
            } else {
                layers_left.last().unwrap().output.as_ref().unwrap()
            };

            let weight_derivative = layer.calc_output_weight_derivatives(layer_input, &backprop_errors, &activation_derivative)?;
            // Update the weights for that layer
            layer.update_weights( 0.8 , &weight_derivative );

            // Calc the error to backproporgate
            backprop_errors = layer.calc_backprop_errors(backprop_errors, activation_derivative);
        }

        Ok(())
    } 
}

impl Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.

        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "Layer {}, inputs: {}, outputs: {}, activation: {:?}", i, layer.n_inputs, layer.n_outputs, layer.activation_function )?;
        }
        
        write!(f,"")
    }
}

pub struct Layer {
    n_inputs: usize,
    n_outputs: usize,
    n_weights: usize,
    weights: Vec<f32>,
    activation_function: Activation,
    activation_inputs: Vec<f32>,
    output: Option<Vec<f32>>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize, activation_function: Activation) -> Layer {
        
        let n_weights = n_inputs * n_outputs;

        let mut weights = Vec::with_capacity( n_weights );
        weights.resize( n_weights , 0.0 );

        let activation_inputs = Vec::with_capacity( n_inputs );
        
        Layer {
            n_inputs,
            n_outputs,
            n_weights,
            weights,
            activation_function,
            activation_inputs,
            output: None
        }
    }

    pub fn new_with_rand(n_inputs: usize, n_outputs: usize, activation_function: Activation) -> Layer {
        let mut new_layer = Layer::new( n_inputs, n_outputs, activation_function);
        
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

    fn process(&mut self, input: &Vec<f32>) -> Result<(), AIError> {
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

// Function to calc the max of a vector
pub fn calc_vec_max( input: &Vec<f32> ) -> Vec<f32> {
    let mut max: f32 = -1.0;
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
pub fn normalise( input: &Vec<f32>, input_min: f32, input_max: f32, output_min: f32, output_max: f32 ) -> Vec<f32> {
    // Create the vector
    let mut output: Vec<f32> = Vec::new();

    for val in input {
        output.push( lin_interp(*val, input_min, input_max, output_min, output_max) );
    }

    // return the output vector
    output
}

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