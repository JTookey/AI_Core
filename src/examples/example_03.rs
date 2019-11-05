use ai_core::*;

fn main() {
    // Create Need-Action NN
    let mut nn_action = NetworkBuilder::new(1)
        .add_layer(3, Activation::Sigmoid)
        .add_layer(3, Activation::Sigmoid)
        .build().expect("Oops");
    let mut action_out: Vec<f32> = Vec::new();

    println!("{}",nn_action);
    
    // Create Need-Situation NN
    let mut nn_situ = NetworkBuilder::new(1)
        .add_layer(3, Activation::Sigmoid)
        .add_layer(3, Activation::Sigmoid)
        .build().expect("Oops");
    let mut situ_out: Vec<f32> = Vec::new();

    println!("{}",nn_situ);

    // Create Need-Situation NN
    let mut nn_attention = NetworkBuilder::new(1)
        .add_layer(3, Activation::Sigmoid)
        .add_layer(3, Activation::Sigmoid)
        .build().expect("Oops");
    let mut atten_out: Vec<f32> = Vec::new();
    println!("{}",nn_attention);

    // Create Delta-Action NN
    let mut nn_delta_action = NetworkBuilder::new(3)
        .add_layer(3, Activation::Sigmoid)
        .add_layer(3, Activation::Sigmoid)
        .build().expect("Oops");
    let mut delta_action_out: Vec<f32> = Vec::new();

    println!("{}",nn_delta_action);


    // Input vector (our need e.g. to eat)
    let input: Vec<f32> = vec![1.0];
    
    // Actual situation (e.g. sensory input)
    let situ_v = vec![
        vec![0.8  , 0.10, 0.6],
        vec![0.75 , 0.12, 0.1],
        vec![0.82 , 0.09, 0.8],
        vec![0.78 , 0.12, 0.5],
        vec![0.81 , 0.11, 0.2],
        ];
     
    for i in 0..1000 {
        let s_idx = i % situ_v.len();
        println!("Actual Situ{:.3?}\n", situ_v[s_idx]);

        // Get the outputs of the NNs
        match nn_action.feedforward(&input) {
            Ok(output) => {
                action_out = output.clone();
                println!("Raw Action {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }

        match nn_situ.feedforward(&input) {
            Ok(output) => {
                situ_out = output.clone();
                println!("Raw Situ   {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }

        match nn_attention.feedforward(&input) {
            Ok(output) => {
                atten_out = output.clone();
                println!("Raw Atten  {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }
        println!("Attention  {:.3?}", atten_out);

        let mut si_a = multiply_vec(&situ_out, &atten_out).unwrap();
        let mut sa_a = multiply_vec(&situ_v[s_idx], &atten_out).unwrap();

        println!("Multi Si*A {:.3?}", si_a);
        println!("Multi Sa*A {:.3?}", sa_a);

        let mut s_diff = calc_diff(&sa_a, &si_a).unwrap();
        let mut sit_dist = calc_length(&s_diff);
        let mut suppresed = calc_suppression(&action_out, &sit_dist);
        println!("Diff       {:.3?}, len = {:.3}", s_diff, sit_dist);
        println!("Suppressed {:.3?}", suppresed);

        match nn_delta_action.feedforward(&s_diff) {
            Ok(output) => {
                delta_action_out = output.clone();
                println!("Delta Act  {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }
        let mut output = add_vector(&suppresed, &delta_action_out).unwrap();
        println!("Output     {:.3?}", output);

        let at_learn = calc_attention(&situ_out, &situ_v[s_idx]).unwrap();

        println!("\nAtten Learn{:.3?}", at_learn);

        // Selected
        let selected = calc_vec_max(&output);
        if selected[0] == 1.0 { // Action Supported Need
            println!("Action supported need....");
            if let Err(e) = nn_action.backproporgate(&input, &selected){ // reinforce action taken
                eprintln!("{}",e);  
            }
            if let Err(e) = nn_situ.backproporgate(&input, &situ_v[s_idx]){
                eprintln!("{}",e);  
            }
            if let Err(e) = nn_attention.backproporgate(&input, &at_learn){
                eprintln!("{}",e);  
            }

        } else if selected[2] == 1.0 { // Action reduced delta
            println!("Action reduced situation error....");
            if let Err(e) = nn_delta_action.backproporgate(&s_diff, &selected){
                eprintln!("{}",e);  
            }
        
        } else { // Action did neither
            println!("Action didn't help....");
            let inv_selected = sub_f_vector(&selected, 1.0).unwrap();
            if let Err(e) = nn_delta_action.backproporgate(&s_diff, &inv_selected){
                eprintln!("{}",e);  
            }
        }
        println!();

        // Repeat calcs
        // Get the outputs of the NNs
        match nn_action.feedforward(&input) {
            Ok(output) => {
                action_out = output.clone();
                println!("Raw Action {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }

        match nn_situ.feedforward(&input) {
            Ok(output) => {
                situ_out = output.clone();
                println!("Raw Situ   {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }

        match nn_attention.feedforward(&input) {
            Ok(output) => {
                atten_out = output.clone();
                println!("Raw Atten  {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }
        println!("Attention  {:.3?}", atten_out);

        si_a = multiply_vec(&situ_out, &atten_out).unwrap();
        sa_a = multiply_vec(&situ_v[s_idx], &atten_out).unwrap();

        println!("Multi Si*A {:.3?}", si_a);
        println!("Multi Sa*A {:.3?}", sa_a);

        s_diff = calc_diff(&sa_a, &si_a).unwrap();
        sit_dist = calc_length(&s_diff);
        suppresed = calc_suppression(&action_out, &sit_dist);
        println!("Diff       {:.3?}, len = {:.3}", s_diff, sit_dist);
        println!("Suppressed {:.3?}", suppresed);

        match nn_delta_action.feedforward(&s_diff) {
            Ok(output) => {
                delta_action_out = output.clone();
                println!("Delta Act  {:.3?}", output )
            },
            Err(e) => eprintln!("{}", e),
        }
        output = add_vector(&suppresed, &delta_action_out).unwrap();
        println!("Output     {:.3?}", output);
    }
}

fn calc_attention(situ_i: &Vec<f32>, situ_a: &Vec<f32>) -> Result<Vec<f32>, AIError> {
    
    if situ_i.len() != situ_a.len() {
        return Err(AIError::LengthMismatch);
    }
    
    let mut out: Vec<f32> = Vec::new();

    for i in 0..situ_i.len() {
        out.push( 1.0 - (situ_i[i] - situ_a[i]).abs() );
    }

    Ok(out)
}

fn calc_diff(in_1: &Vec<f32>, in_2: &Vec<f32>) -> Result<Vec<f32>, AIError> {
    
    if in_1.len() != in_2.len() {
        return Err(AIError::LengthMismatch);
    }
    
    let mut out: Vec<f32> = Vec::new();

    for i in 0..in_1.len() {
        out.push( in_1[i] - in_2[i] );
    }

    Ok(out)
}

fn calc_length(v_in: &Vec<f32> ) -> f32 {
    let mut sum: f32 = 0.0;
    for elem in v_in {
        sum += elem * elem;
    }
    sum.sqrt()
}

fn calc_suppression( action_vec: &Vec<f32>, sit_dist: &f32 ) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();

    for elem in action_vec {
        out.push( elem * (1.0 - sit_dist*sit_dist ) );
    }

    out
}

fn add_vector( in_1: &Vec<f32>, in_2: &Vec<f32>) -> Result<Vec<f32>, AIError> {
    if in_1.len() != in_2.len() {
        return Err(AIError::LengthMismatch);
    }
    
    let mut out: Vec<f32> = Vec::new();
    for i in 0..in_1.len() {
        out.push( in_1[i] + in_2[i] );
    }

    Ok(out)
}

fn sub_f_vector( in_1: &Vec<f32>, in_2: f32) -> Result<Vec<f32>, AIError> {
    let mut out: Vec<f32> = Vec::new();
    for i in 0..in_1.len() {
        out.push( in_2 - in_1[i] );
    }

    Ok(out)
}