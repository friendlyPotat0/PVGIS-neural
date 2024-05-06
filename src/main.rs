mod interface;

use std::io;
use interface::Interface;

fn main() {
    println!("PVGIS NEURAL\n1. Train model\n2. Print power given timestamp\n3. Save to disk power profile (year)\n4. Save to disk calculated energy from timestamp range\n5. Quit");
    let option = request_num_input(&"Enter option: ".to_string());
    match option {
          1=>Interface::train_neural_network(),
          2=>Interface::print_result_from_loaded(),
          3=>Interface::write_result_from_loaded_given_input_range(),
          4=>Interface::integrate_from_loaded_given_input_range_and_write(),
          5=>println!("Bye!"),
          _=>println!("Not a valid option. Try again"),
    }
}

fn request_num_input(message: &String) -> i32 {
    let mut num = String::new();
    print!("{}", message);
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut num)
        .expect("Failed to read line");
    let num: i32 = num.trim().parse().expect("Input must be an integer");
    return num;
}

fn request_string_input(message: &String) -> String {
    let mut num = String::new();
    print!("{}", message);
    io::Write::flush(&mut io::stdout()).expect("Failed to flush stdout");
    io::stdin()
        .read_line(&mut num)
        .expect("Failed to read line");
    // let num: i32 = num.trim().parse().expect("Input must be an integer");
    return num;
}
