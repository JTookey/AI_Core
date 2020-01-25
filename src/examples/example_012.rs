use ndarray::{arr1, Array, Array1, Array2, Zip, linalg};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ai_core::util::normalise;

fn main() {
    println!("Cyril: I think and therefore I am");

    // Create the input and the expected output
    let mut input: Array1<f64> = arr1( &[10.0, 10.0, 10.0, 10.0]  );
    let out_expected: Array1<f64> = arr1( &[0.2, 0.8] );

    // Demonstrate the normalising function
    normalise(&mut input, 10.0, 20.0, 0.0, 1.0);
    println!("Input Vector: {:.3}", input );
    println!("Output Expected: {:.3}\n", out_expected);


    // Create weights between 1 and -1
    let mut w = Array::random((2, 4), Uniform::new( -1.0, 1.0 ));
    println!("Weights: {:.3}", w );
    // Create bias array initially 0
    let mut b = arr1( &[0.0, 0.0] );
    println!("Bias: {:.3}", b );


    // Create the required arrays
    let mut act_in: Array1<f64> = Array1::zeros( w.nrows() );
    let mut output: Array1<f64> = arr1( &[0.0, 0.0] );
    let mut error: Array1<f64> = arr1( &[0.0 , 0.0]);
    let mut total_error: f64; // unassigned because initial value immediately overwritten
    let mut act_der: Array1<f64> = arr1( &[0.0, 0.0] );
    let mut b_der: Array1<f64> = Array::zeros( 2 );
    let mut w_der: Array2<f64> = Array::zeros( (2, 4) );
    let learning_rate: f64 = 0.8;
    
    for i in 0..50 {
        println!("\nIteration... {}", i);

        // Create the Output!!!

        // Step 1 - Calculate the Activation Input
        Zip::from(&mut act_in).apply(|x| *x = 0.0);
        linalg::general_mat_vec_mul(1.0, &w, &input, 1.0, &mut act_in);
        act_in += &b;
        println!("Activation Input: {:.3}", act_in );

        // Step 2 - Calculate the Activation (i.e. the output) for Sigmoid
        Zip::from( &mut output )
            .and( &act_in )
            .apply( | output, &input |  *output = sigmoid(input) );
        println!("Activation Output: {:.3}", output );


        // Time to LEARN!!!

        // Step 1 - Calculate the Output Error
        Zip::from( &mut error )
            .and( &output )
            .and( &out_expected )
            .apply( |e, o, o_e | *e = o - o_e );
        total_error = error.iter().fold(0.0, | acc, elm | acc + elm.powi(2)) / output.len() as f64;
        println!("Total Error: {:.6}", total_error);
        println!("\nError: {:.3}", error);

        // Step 2 - Calculate the Activation Derivative (length of activation input)
        Zip::from( &mut act_der)
            .and( &act_in )
            .apply( | a_d, &a_i | *a_d = sigmoid_derivative( a_i ) );
        println!("Activation Derivative: {:.3}", act_der);

        // Step 3 - Calculate the Bias Derivative
        Zip::from( &mut b_der )
            .and( &act_der )
            .and( &error )
            .apply( |b_der, &act_der, &error| *b_der = act_der*error );
        println!("Bias Derivative: {:.3}", b_der);
        
        // Step 4 - Calculate the Weight Derivative - (reusing b_der as this is act_d * error)
        for (i, b_d) in b_der.iter().enumerate() {
            for (j, inp) in input.iter().enumerate() {
                w_der[[i,j]] = b_d * inp * learning_rate;
            }
        }
        println!("Weight Derivative: {:.3}", w_der);

        // Step 5 - Update Weights
        w -= &w_der;
        println!("\nUpdated Weights: {:.3}",w);
        
        b.zip_mut_with( &b_der, | b, &der | *b -= der * learning_rate );
        println!("Updated Bias: {:.3}", b);
    }
}

// Sigmoid Function - single value
fn sigmoid(input: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * input).exp())
}

// Sigmoid Derivative - single value
fn sigmoid_derivative(input: f64) -> f64 {
    let s = sigmoid(input);
    s * (1.0 - s)
}