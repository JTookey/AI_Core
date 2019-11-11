/// Basic utility functions

use rand::prelude::*;

/// Create vector with random numbers
pub fn new_random_vec<T>( len: usize ) -> Vec<T> 
    where rand::distributions::Standard: rand::distributions::Distribution<T>
{
    let mut out: Vec<T> = Vec::with_capacity( len );
    let mut rng = rand::thread_rng();

    for _ in 0..len {
        out.push( rng.gen() );
    }

    // return
    out
}

/// Function to linearly interpolate
pub fn lin_interp<T>( input: T, input_min: T, input_max: T, output_min: T, output_max: T ) -> T 
    where T
        : std::ops::Add<Output = T> 
        + std::ops::Sub<Output = T> 
        + std::ops::Mul<Output = T> 
        + std::ops::Div<Output = T> 
        + Copy
{
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