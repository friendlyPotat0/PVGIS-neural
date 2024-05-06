extern crate neuroflow;

use chrono::prelude::*;
use rand::Rng;

use neuroflow::activators::Type::Tanh;
use neuroflow::data::DataSet;
use neuroflow::FeedForward;

pub struct Interface {}

impl Interface {
    pub fn train_test() {
        let mut nn = FeedForward::new(&[2, 2, 1]);
        let logfile = String::from("resources/rmse_00.txt");
        nn.set_logfile(&logfile);
        let mut data: DataSet = DataSet::new();
        let mut i = -3.0;
        let mut rng = rand::thread_rng();
        while i <= 2.5 {
            let a: u8 = rng.gen_range(0..=1);
            let b: u8 = rng.gen_range(0..=1);
            data.push(&[a.into(), b.into()], &[(a ^ b).into()]);
            i += 0.05;
        }
        nn.activation(Tanh)
            .learning_rate(0.1)
            .momentum(0.15)
            .train(&data);
        let mut res;
        i = 0.0;
        while i <= 0.3 {
            let a: u8 = rng.gen_range(0..=1);
            let b: u8 = rng.gen_range(0..=1);
            res = nn.calc(&[a.into(), b.into()])[0];
            println!(
                "for [{:.3}], [{:.3}] = [{:.3}] -> [{:.3}]",
                a,
                b,
                a ^ b,
                res
            );
            i += 0.07;
        }
    }
    pub fn train_neural_network() {
        // Get current UTC time
        let utc: DateTime<Utc> = Utc::now();
        let unix_timestamp = utc.timestamp();
        let unix_timestamp_str = unix_timestamp.to_string();
        // ---
        let mut samples = 0; // this value change it
                             // create neural network and specify topology
        let mut nn = FeedForward::new(&[3, 64, 32, 1]);
        // define logfile for RMSE evolution
        let filename = format!(
            "resources/errors/rmse_{}_samples_{}.txt",
            samples, unix_timestamp_str
        );
        nn.set_logfile(&filename);
        // define training data set
        let mut data: DataSet = DataSet::new();
        // data.push(&[a.into(), b.into()], &[(a ^ b).into()]);
        // define neural network architecture
        nn.activation(Tanh)
            .learning_rate(0.1)
            .momentum(0.15)
            .train(&data);
        // save model to disk
        let filename = format!(
            "resources/models/model_{}_samples_{}.flow",
            samples, unix_timestamp_str
        );
        neuroflow::io::save(&mut nn, &filename).unwrap();
    }

    pub fn print_result_from_loaded() {}

    pub fn write_result_from_loaded_given_input_range() {}

    pub fn integrate_from_loaded_given_input_range_and_write() {}
}
