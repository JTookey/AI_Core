use ndarray::{arr1, Array1, Zip};
use ai_core::util::normalise;
use ai_core::layer::*;

fn main() {
    println!("Cyril: I think and therefore I am");

    // Create the input and the expected output
    let mut input: Array1<f64> = arr1( &[10.0, 20.0, 10.0, 10.0]  );
    let out_expected: Array1<f64> = arr1( &[0.2, 0.8] );

    // Demonstrate the normalising function
    normalise(&mut input, 10.0, 20.0, 0.0, 1.0);
    println!("Input Vector: {:.3}", input );
    println!("Output Expected: {:.3}\n", out_expected);

    // Create the layer
    let mut layer = BaseLayer::new( 4, 2, Activation::Sigmoid );

    // Create the required arrays
    let mut output: Array1<f64> = Array1::zeros( 2 );
    let mut error: Array1<f64> = Array1::zeros( 2 );
    let mut backprop_error: Array1<f64> = Array1::zeros( 4 );
    let mut total_error: f64;
    let learning_rate: f64 = 0.8;
    
    for i in 0..50 {
        println!("\nIteration... {}", i);

        // Create the Output!!!

        layer.feedforward(&input, &mut output);
        println!("Output: {:.3}", output);

        // Time to LEARN!!!

        // Step 1 - Calculate the Output Error
        Zip::from( &mut error )
            .and( &output )
            .and( &out_expected )
            .apply( |e, o, o_e | *e = o - o_e );
        total_error = error.iter().fold(0.0, | acc, elm | acc + elm.powi(2)) / output.len() as f64;
        println!("Error: {:.3}", error);
        println!("Total Error: {:.6}", total_error);
        
        // Step 2 - Backproporgate
        layer.backproporgate(&input, &error, &mut backprop_error, learning_rate);


        // Need to calculate the backproporgated errors. Can't re-use the same error 
        // array because it may need to change size depending on the number of nodes 
        // in the layer
    }
}