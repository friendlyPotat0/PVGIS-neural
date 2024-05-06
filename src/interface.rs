extern crate neuroflow;

use std::fmt::format;
use chrono::prelude::*;

use neuroflow::activators::Type::Tanh;
use neuroflow::data::DataSet;
use neuroflow::FeedForward;

pub struct Interface {
}

impl Interface {
    pub fn train_neural_network() {
        // Get current UTC time
        let utc: DateTime<Utc> = Utc::now();
        let unix_timestamp = utc.timestamp();
        let unix_timestamp_str = unix_timestamp.to_string();
        // ---
        let mut epochs = 10; // this value change it
        // create neural network and specify topology
        let mut nn = FeedForward::new(&[3, 64, 32, 1]);
        // define logfile for RMSE evolution
        let filename = format!("resources/errors/rmse_{}_epochs_{}.txt", epochs, unix_timestamp_str);
        nn.set_logfile(&filename);
        // define training data set
        let mut data: DataSet = DataSet::new();
        // data.push(&[a.into(), b.into()], &[(a ^ b).into()]);
        // define neural network architecture
        nn.activation(Tanh)
            .learning_rate(0.1)
            .momentum(0.15)
            .train(&data, 2500);
        // save model to disk
        let filename = format!("resources/models/model_{}_epochs_{}.flow", epochs, unix_timestamp_str);
        neuroflow::io::save(&mut nn, &filename).unwrap();
    }

    pub fn print_result_from_loaded() {}

    pub fn write_result_from_loaded_given_input_range() {}

    pub fn integrate_from_loaded_given_input_range_and_write() {}

}
