mod interface;

use interface::Interface;
use neuroflow::activators::Type;
use std::{env, io};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    println!("PVGIS NEURAL");
    println!("1. Train neural network");
    println!("2. Predict power profile from time range (Unix Epoch)");
    println!("3. Compute energy from profile");
    println!("4. Quit");
    let option = request_num_input("Enter option: ");
    match option {
        0 => Interface::train_test(),
        1 => {
            let database_path = request_string_input("Enter path to database: ");
            let mut interface = Interface::new(database_path);
            let output_path = request_string_input("Enter path to store trained model: ");
            let log_file = request_string_input("Enter log file path [optional]: ");
            interface.enable_error_logging(log_file);
            let samples = request_num_input("Enter number of samples to feed [0 for all available]: ");
            let topology = request_int_slice_input("Enter topology: ");
            let activation_function = request_activation_function();
            let eta = request_double_input("Enter learning rate: ");
            let alpha = request_double_input("Enter momentum: ");
            interface.train_neural_network(
                &output_path,
                samples as usize,
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

fn request_num_input(message: &str) -> i32 {
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

fn request_double_input(message: &str) -> f64 {
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
        Ok(1) => Type::Tanh,
        Ok(2) => Type::Relu,
        Ok(3) => Type::Sigmoid,
        _ => {
            println!("Invalid choice. Defaulting to Tanh");
            Type::Tanh
        }
    }
}
