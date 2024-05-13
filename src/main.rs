mod trainer;

use trainer::Trainer;
use neuroflow::activators::Type;
use std::{env, io};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    println!("PVGIS NEURAL");
    println!("0. Synthesize data");
    println!("1. Train neural network");
    println!("2. Predict power profile from time range (Unix Epoch)");
    println!("3. Compute energy from profile");
    println!("4. Quit");
    let option = request_i32_input("Enter option: ");
    match option {
        0 => {
            let database_path = request_string_input("Enter path to database: ");
            let trainer = Trainer::new(database_path.clone());
            let output_path = request_string_input("Enter path to store synthesized database: ");
            trainer.synthesize_pvgis_data(&database_path, &output_path);
        }
        1 => {
            let database_path = request_string_input("Enter database file: ");
            let mut trainer = Trainer::new(database_path);
            let output_path = request_string_input("Enter path to store trained model: ");
            let log_file = request_string_input("Enter log file path [optional]: ");
            if !log_file.is_empty() {
                let epoch_interval = request_i32_input("Enter epoch interval: ");
                trainer.enable_error_logging(log_file, epoch_interval);
            }
            let topology = request_int_slice_input("Enter topology: ");
            let activation_function = request_activation_function();
            let eta = request_f64_input("Enter learning rate: ");
            let alpha = request_f64_input("Enter momentum: ");
            trainer.train_neural_network(
                &output_path,
                &topology,
                activation_function,
                eta,
                alpha,
            );
        }
        // 2 => Interface::print_result_from_loaded(),
        // 3 => Interface::write_result_from_loaded_given_input_range(),
        // 4 => Interface::integrate_from_loaded_given_input_range_and_write(),
        4 => println!("Bye!"),
        _ => println!("Not a valid option. Try again"),
    }
}

fn request_i32_input(message: &str) -> i32 {
    let mut input = String::new();
    print!("{}", message);
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    input.trim().parse().expect("Input must be an integer")
}

fn request_string_input(message: &str) -> String {
    let mut input = String::new();
    print!("{}", message);
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    input.trim().to_string()
}

fn request_f64_input(message: &str) -> f64 {
    let mut input = String::new();
    print!("{}", message);
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    input.trim().parse().expect("Input must be an integer")
}

fn request_int_slice_input(message: &str) -> Vec<i32> {
    let mut input = String::new();
    print!("{}", message);
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    input
        .trim()
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect()
}

fn request_activation_function() -> Type {
    println!("Select an activation function:");
    println!("0. ELU");
    println!("1. Tanh");
    println!("2. Relu");
    println!("3. Sigmoid");
    let mut input = String::new();
    print!("Enter choice: ");
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    match input.trim().parse::<usize>() {
        Ok(0) => Type::ELU,
        Ok(1) => Type::Tanh,
        Ok(2) => Type::Relu,
        Ok(3) => Type::Sigmoid,
        _ => {
            println!("Invalid choice. Defaulting to ELU");
            Type::ELU
        }
    }
}
