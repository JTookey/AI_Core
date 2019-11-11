// Use Statements
use std::fmt::{self, Display};
use rand::prelude::*;
use crate::err::AIError;
use crate::layer::*;

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