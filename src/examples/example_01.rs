use ai_core::layer::*;
use ai_core::util::calc_average_sum_square;

fn main() {
    println!("I think and therefore I am!\n");

    // Create simple net
    let input: Vec<f32> = vec![1.0, 0.0];
    let out_expected = vec![0.2, 0.8];

    // Normalise
    let norm_input = normalise( &input , 0.0 , 1.0, 0.0, 1.0);
    println!("Input Vector: {:.3?}",norm_input);
    println!("Output Expected: {:.3?}\n", out_expected);


    // Create a layer
    let mut first_layer = Layer::new_with_rand( 2, 3, Activation::Sigmoid );
    let mut second_layer = Layer::new_with_rand( 3, 2, Activation::Sigmoid );

    let mut run = true;
    let mut run_count = 0;

    let mut output: Vec<f32> = Vec::new();
    let mut current_error: f32 = 0.0;

    while run {
        // Generate Output
        let activation_inputs_first = first_layer.gen_activation_inputs( &norm_input ).unwrap();
        let output_first = activation_sigmoid( &activation_inputs_first );
        let activation_inputs_second = second_layer.gen_activation_inputs( &output_first ).unwrap();
        output = activation_sigmoid( &activation_inputs_second );

        // Learn
        let out_error = calc_output_layer_error( &output, &out_expected).unwrap();
        current_error = calc_average_sum_square(&out_error);

        let activation_derivative_second = derivative_sigmoid(&activation_inputs_second);
        let weight_derivative_second = second_layer.calc_output_weight_derivatives(&output_first, &out_error, &activation_derivative_second).unwrap();
        let backprop_errors_second = second_layer.calc_backprop_errors(out_error, activation_derivative_second);

        let activation_derivative_first = derivative_sigmoid(&activation_inputs_first);
        let weight_derivative_first = first_layer.calc_output_weight_derivatives(&norm_input, &backprop_errors_second, &activation_derivative_first).unwrap();
        
        // Update the weights
        first_layer.update_weights( 0.8 , &weight_derivative_first );
        second_layer.update_weights( 0.8 , &weight_derivative_second );

        // Check if continue running
        run_count += 1;
        if run_count > 1000 || current_error < 0.00001 {
            run = false;
        }
    }
    
    println!("Run Count: {}", run_count);
    println!("Current Error: {}\n", current_error);
    println!("First Layer Weights:\n{}\n", first_layer);
    println!("Second Layer Weights:\n{}\n", second_layer);
    println!("Output Vector: {:.3?}\n", output);
}