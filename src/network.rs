// Use Statements
use std::fmt::{self, Display};
use rand::prelude::*;
use crate::err::AIError;
use crate::layer::*;
use crate::{AIVec, AIWeights};

use ndarray::Zip;


/// Network Builder
/// 
/// This is the main way to construct a network. It provides an easy interface for the user:
/// NetworkBuilder::new( n_inputs ).add_layer( n_nodes )
/// 
/// Note that the outputs are equal to the number of nodes in last layer
pub struct NetworkBuilder{
    n_inputs: usize,
    n_outputs: usize,
    layers: Vec<(usize, usize, Activation)>,
    feedforward_inputs: Vec<usize>,
    backprop_errors: Vec<usize>,
}

// Implement the function interface
impl NetworkBuilder {

    /// Creates a zero layer network with only the number of inputs set
    /// 
    /// Note this expects you to add at least one layer to set the number of outputs
    pub fn new(n_inputs: usize) -> NetworkBuilder {
        NetworkBuilder{
            n_inputs,
            n_outputs: 0,
            layers: Vec::new(),
            feedforward_inputs: Vec::new(),
            backprop_errors: Vec::new(),
        }
    }

    /// Function to add another layer to the feedforward network
    pub fn add_layer(&mut self, n_nodes: usize, activation_function: Activation) -> &mut Self {
        if self.layers.len() == 0 {
            // If no layers then add nodes/weights based on the number of network inputs
            self.layers.push( (self.n_inputs, n_nodes, activation_function) );
        } else {
            // If there is already layers in the network then add nodes/weights to the 
            // new layer based on the number of outputs from the previous layer
            let last_n_outputs = self.layers.last().unwrap().1;
            self.layers.push( (last_n_outputs, n_nodes, activation_function) );

            // If there are now multiple layers we also need to add vectors for holding the feedforward input buffer
            self.feedforward_inputs.push( last_n_outputs );
        }

        // add to the backprop errors
        self.backprop_errors.push( n_nodes );
        
        // set the number of outputs from the network equal to the nodes in the last layer
        self.n_outputs = n_nodes;

        self
    }

    /// Build will construct and return the actual network
    pub fn build(&self) -> Option<NeuralNetwork> {

        let mut n_layers: usize = 0;

        // Create the layers
        let mut layers: Vec<BaseLayer> = Vec::new();
        for (n_inputs, n_outputs, activation_function) in &self.layers {
            layers.push( BaseLayer::new(*n_inputs, *n_outputs, activation_function.clone()) );
            n_layers += 1;
        }

        // Create the feedforward inputs
        let mut feedforward_inputs: Vec<AIVec> = Vec::new();
        for s in &self.feedforward_inputs {
            feedforward_inputs.push( AIVec::zeros( *s ) );
        }

        // Create the backprop errors
        let mut backprop_errors: Vec<AIVec> = Vec::new();
        for s in &self.backprop_errors {
            backprop_errors.push( AIVec::zeros( *s ) );
        }

        // Create the NeuralNetwork struct
        Some(NeuralNetwork {
            n_inputs: self.n_inputs,
            n_outputs: self.n_outputs,
            n_layers,
            layers,
            feedforward_inputs,
            backprop_errors,
            last_input: None,
        })
    }
}

/// The Feedforward NeuralNetwork structure that contains all of the layers and neurons
pub struct NeuralNetwork {
    n_inputs: usize,
    n_outputs: usize,
    n_layers: usize,
    layers: Vec<BaseLayer>,
    feedforward_inputs: Vec<AIVec>,
    backprop_errors: Vec<AIVec>,
    last_input: Option<AIVec>,
}

impl NeuralNetwork {

    fn number_of_inputs(&self) -> usize {
        self.n_inputs
    }

    fn number_of_outputs(&self) -> usize {
        self.n_outputs
    }

    fn number_of_layers(&self) -> usize {
        self.n_layers
    }

    fn layers(&self) -> &Vec<BaseLayer> {
        &self.layers
    }
    
    // TODO: Decide if we even want this function
    fn check_input(&self, input: &AIVec) -> Result<(), AIError> {

        // Check there was a last value
        if let Some(l_in) = &self.last_input {
            // Check the length
            if l_in.len() != input.len() {
                return Err(AIError::LengthMismatch);
            }
            // Check values
            if input != l_in {
                return Err(AIError::InputMismatch);
            }
        } else {
            return Err(AIError::Unprocessed);
        }

        Ok(())
    }

    pub fn feedforward(&mut self, input: &AIVec, output: &mut AIVec) -> Result<(), AIError> {
        // Loop through the layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == 0 {
                if self.n_layers == 1 {
                    // If single layer network then directly use input to output
                    layer.feedforward(input, output);
                } else {
                    // If multi layer network then save the inputs to the feedforward inputs buffer
                    layer.feedforward(input, &mut self.feedforward_inputs[i]);
                }
            } else {
                if i < (self.n_layers - 1) {
                    // If not the final layer then send outputs to the next feedforward buffer
                    let (head, tail) = self.feedforward_inputs.split_at_mut(i);
                    let inputs: &AIVec = head.last().unwrap();
                    let output: &mut AIVec = tail.first_mut().unwrap();
                    layer.feedforward( inputs, output );
                } else {
                    // If it is the final layer then outputs to the output of the function
                    layer.feedforward(&self.feedforward_inputs[i-1], output);
                }
            } 
        }

        Ok(())
    }

    pub fn backproporgate(&mut self, input: &AIVec, out_expected: &AIVec) -> Result<(), AIError> {
        // Step 1 - Feedforward using input
        let mut output: AIVec = AIVec::zeros( self.n_outputs ); // TODO: remove this allocation
        self.feedforward(input, &mut output).unwrap();

        // Step 2 - Calculate the Output Error
        Zip::from( self.backprop_errors.last_mut().unwrap() )
            .and( &output )
            .and( out_expected )
            .apply( |e, o, o_e | *e = o - o_e );

        // Step 3 - Backprop through layers in reverse order
        //
        // error[2] + output         -> Layer[2] -> backprop[1]
        // error[1] + feedforward[1] -> Layer[1] -> backprop[0]
        // error[0] + feedforward[0] -> Layer[0] -> unused_backprop

        let mut unused_backprop_err: AIVec = AIVec::zeros( self.layers.first().unwrap().get_n_inputs() ); // TODO: remove this allocation
        
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let ri = self.n_layers - i - 1;
            let (head, tail) = self.backprop_errors.split_at_mut(ri);
            let error: &AIVec = tail.first().unwrap();
            //println!("{}", error);

            if ri == 0 {
                // TODO: Implement alternate backproporgation function for first layer 
                //  that does not generate backprop_errors 
                layer.backproporgate(&input, error, &mut unused_backprop_err, 0.8 ); 
            } else {
                let backprop_error: &mut AIVec = head.last_mut().unwrap();
                layer.backproporgate(&self.feedforward_inputs[ri-1], error, backprop_error, 0.8 );
            }
            
        }

        Ok(())
    }

    pub fn calculate_error(&mut self, input: &AIVec, out_expected: &AIVec) -> Result<f64, AIError> {
        // Step 1 - Feedforward using input
        let mut output: AIVec = AIVec::zeros( self.n_outputs ); // TODO: remove this allocation
        self.feedforward(input, &mut output).unwrap(); // Handle Error properly

        // Step 2 - Calculate the Output Error
        let mut error: AIVec = AIVec::zeros( self.n_outputs ); // TODO: remove this allocation
        Zip::from( &mut error )
            .and( &output )
            .and( out_expected )
            .apply( |e, o, o_e | *e = o - o_e );

        // Step 2 - Calc the sum squared
        let total_error = error.iter().fold(0.0, | acc, elm | acc + elm.powi(2)) / output.len() as f64;

        // Return
        Ok(total_error)
    }
}

impl Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.

        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "Layer {}, inputs: {}, outputs: {}, activation: {:?}", i, layer.get_n_inputs(), layer.get_n_outputs(), layer.get_activation() )?;
        }
        
        write!(f,"")
    }
}