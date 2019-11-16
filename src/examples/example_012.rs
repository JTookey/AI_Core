use ndarray::{arr1, arr2, Array,  Array1, Array2, ArrayBase, OwnedRepr, Ix1, Ix2, Zip, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ai_core::util::normalise;

fn main() {
    println!("Cyril: I think and therefore I am");

    // Create the input and the expected output
    let mut input: Array1<f64> = arr1( &[8.0, 15.0, 18.0, 27.0]  );
    let out_expected: Array1<f64> = arr1( &[0.0, 0.2, 0.8, 1.0] );

    // Demonstrate the normalising function
    normalise(&mut input, 10.0, 20.0, 0.0, 1.0);
    println!("Input Vector: {:.3}", input );
    println!("Output Expected: {:.3}\n", out_expected);

    //let mut w: Array2<f64> = arr2(&[[1.0, 1.0, 1.0, 1.0],
    //                        [2.0, 2.0, 2.0, 2.0]]);

    let mut w = Array::random((2, 5), Uniform::new(0., 10.));
    println!("Weights: {:.3 }", w );

    let mut res: Array1<f64> = Array1::zeros( w.nrows() );
    println!("{}", res );

    // Dot Product Layers

    
    //println!("{}", res );

    // I don't think Zip likes Array1....?

}