extern crate neuroflow;

use std::fs;

use chrono::prelude::*;
use rand::Rng;

use neuroflow::activators::Type::Tanh;
use neuroflow::data::DataSet;
use neuroflow::FeedForward;
use serde_json::Value;

pub struct Interface {}

impl Interface {
    pub fn train_test() {
        let mut nn = FeedForward::new(&[2, 2, 1]);
        let logfile = String::from("resources/errors/rmse_00.txt");
        nn.set_logfile(&logfile);
        let mut data: DataSet = DataSet::new();
        let mut i = -3.0;
        let mut rng = rand::thread_rng();
        while i <= 120.5 {
            let a: u8 = rng.gen_range(0..=1);
            let b: u8 = rng.gen_range(0..=1);
            data.push(&[a.into(), b.into()], &[(a ^ b).into()]);
            i += 0.05;
        }
        nn.activation(Tanh)
            .learning_rate(0.1)
            .momentum(0.15)
            .train(&data);
        neuroflow::io::save(&mut nn, "resources/models/test.flow").unwrap();
        let mut new_nn: FeedForward = neuroflow::io::load("resources/models/test.flow").unwrap();
        let mut res;
        i = 0.0;
        while i <= 0.3 {
            let a: u8 = rng.gen_range(0..=1);
            let b: u8 = rng.gen_range(0..=1);
            res = new_nn.calc(&[a.into(), b.into()])[0];
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
    pub fn train_neural_network(samples: usize) {
        // Get current UTC time
        let utc: DateTime<Utc> = Utc::now();
        let unix_timestamp = utc.timestamp();
        let unix_timestamp_str = unix_timestamp.to_string();

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

        // read files from solar-radiation-database
        let database_dir = "resources/solar-radiation-database";
        let files = fs::read_dir(database_dir).unwrap();

        // Iterate over files
        for file in files {
            if let Ok(file) = file {
                let file_path = file.path();
                if let Some(file_name) = file_path.file_name() {
                    if let Some(file_name_str) = file_name.to_str() {
                        println!("Interpreting: {}!", file_name_str);
                        // Load JSON file
                        let json_str = fs::read_to_string(file_path).unwrap();
                        let json_data: Value = serde_json::from_str(&json_str).unwrap();

                        // Extract latitude and longitude
                        let latitude = json_data["inputs"]["location"]["latitude"]
                            .as_f64()
                            .unwrap();
                        let longitude = json_data["inputs"]["location"]["longitude"]
                            .as_f64()
                            .unwrap();

                        // Extract data from the "hourly" array
                        if let Some(hourly_data) = json_data["outputs"]["hourly"].as_array() {
                            for hourly_entry in hourly_data {
                                let timestamp_str = hourly_entry["time"].as_str().unwrap();
                                let power = hourly_entry["G(i)"].as_f64().unwrap();

                                // Convert timestamp to Unix timestamp
                                let timestamp_unix = Self::convert_to_unix_timestamp(timestamp_str);

                                // Normalize latitude and longitude (you need to implement normalization)
                                let latitude_norm = Self::normalize_latitude(latitude);
                                let longitude_norm = Self::normalize_longitude(longitude);

                                // Push data to training set
                                data.push(
                                    &[latitude_norm, longitude_norm, timestamp_unix as f64],
                                    &[power],
                                );
                            }
                        }
                    }
                }
            }
        }

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

    fn normalize_latitude(latitude: f64) -> f64 {
        // Latitude ranges from -90 to 90 degrees.
        // Normalize it to the range [0, 1] or [-1, 1].
        (latitude + 90.0) / 180.0 // Normalize to [0, 1]
    }

    fn normalize_longitude(longitude: f64) -> f64 {
        // Longitude ranges from -180 to 180 degrees.
        // Normalize it to the range [0, 1] or [-1, 1].
        (longitude + 180.0) / 360.0 // Normalize to [0, 1]
    }

    fn convert_to_unix_timestamp(timestamp_str: &str) -> i64 {
        // Parse the timestamp string
        let datetime = NaiveDateTime::parse_from_str(timestamp_str, "%Y%m%d:%H%M").unwrap();
        // Convert to UTC
        let utc_datetime = DateTime::<Utc>::from_utc(datetime, Utc);
        // Convert to Unix timestamp
        let unix_timestamp = utc_datetime.timestamp();
        return unix_timestamp;
    }

    pub fn print_result_from_loaded() {}

    pub fn write_result_from_loaded_given_input_range() {}

    pub fn integrate_from_loaded_given_input_range_and_write() {}
}
