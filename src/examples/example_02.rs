use ai_core::layer::*;
use ai_core::network::*;
use ai_core::AIVec;

use ndarray::arr1;

fn main() {
    // Create a Feedforward NeuralNetwork with 2 inputs, 3 layers and 4 outputs
    // First Layer
    let mut nn = NetworkBuilder::new(2)
        .add_layer(3, Activation::Sigmoid) // First layer has 2 inputs and 3 nodes
        .add_layer(5, Activation::Sigmoid) // Second layer has 3 inputs and 5 nodes
        .add_layer(4, Activation::Sigmoid) // Third layer has 5 inputs and 4 nodes
        .build().expect("Oops");

    // Print the network for debug purposes
    println!("{}",nn);

    // Create the AIVec that will hold the inputs and outputs of the network
    let input: AIVec = arr1( &[0.5, 0.8] );
    let mut output: AIVec = AIVec::zeros( 4 );
    let expected = arr1( &[0.1, 0.1, 0.1, 0.1] );

    // Run the inputs through the network for the first time
    match nn.feedforward(&input, &mut output) {
        Ok(()) => { println!("Output {}", output); },
        Err(e) => { eprintln!("{}", e); },
    };

    // Calculate the error at the start
    println!("Error Before: {}", nn.calculate_error(&input, &expected).unwrap()); 
    
    // Loop through a number of backproporgation cycles
    for i in 0..10 {
        // Carry out backproporgation
        if let Err(e) = nn.backproporgate(&input, &expected){
            eprintln!("{}",e);  
        }
      
        // Recalculate the error
        println!("Error After {}: {:.6}", i, nn.calculate_error(&input, &expected).unwrap()); 
    }
}