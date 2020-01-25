/// Basic utility functions

use std::f64;
use ndarray::Array1;
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

/// Function for normalising a 1D array
pub fn normalise<'a, T>( array: &'a mut Array1<T>, input_min: T, input_max: T, output_min: T, output_max: T )
    where T
        : num_traits::NumOps
        + std::cmp::PartialOrd 
        + Copy + Clone,
        // This needs cleaning up using the num_traits, but for the time being it helps me with clarity
{
    for elem in array.iter_mut() {
        if *elem < input_min {
            *elem = output_min;
        } else if *elem > input_max {
            *elem = output_max;
        } else {
            *elem = lin_interp::<T>(*elem, input_min, input_max, output_min, output_max);
        }
    }
}

/// Function to linearly interpolate
pub fn lin_interp<T>( input: T, input_min: T, input_max: T, output_min: T, output_max: T ) -> T 
    where T
        : num_traits::NumOps
        + Copy
{
    output_min + (output_max - output_min) * (input - input_min) / (input_max - input_min)
}

/// Function to calculate error
pub fn calc_average_sum_square( vector: &Vec<f64> ) -> f64 {
    let mut average_sum_square: f64 = 0.0;

    for index in 0..vector.len() {
        average_sum_square += vector[index].powi(2);
    }

    average_sum_square /= vector.len() as f64;

    // return
    average_sum_square
}

/// Function to calculate the max
pub fn calc_index_of_max( vector: &Array1<f64>) -> usize {
    
    let mut max_found = f64::MIN;
    let mut max_index = 0;

    for (i, v) in vector.iter().enumerate() {
        if v > &max_found {
            max_found = *v;
            max_index = i;
        }
    }

    max_index
}