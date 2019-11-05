use ai_core::*;

fn main() {
    let mut nn = NetworkBuilder::new(2)
        .add_layer(3, Activation::Sigmoid)
        .add_layer(5, Activation::Sigmoid)
        .add_layer(4, Activation::Sigmoid)
        .build().expect("Oops");

    println!("{}",nn);

    let input: Vec<f32> = vec![0.5, 0.8];
    let result = vec![0.1, 0.1, 0.1, 0.1];

    match nn.feedforward(&input) {
        Ok(output) => println!("Output {:?}\nError {}"
            ,output
            ,calc_average_sum_square(&calc_output_layer_error(&output, &result).unwrap())),
        Err(e) => eprintln!("{}", e),
    }
    
    if let Err(e) = nn.backproporgate(&input, &result){
      eprintln!("{}",e);  
    } 

    match nn.feedforward(&input) {
        Ok(output) => println!("Output {:?}\nError {}"
            ,output
            ,calc_average_sum_square(&calc_output_layer_error(&output, &result).unwrap())),
        Err(e) => eprintln!("{}", e),
    }
}