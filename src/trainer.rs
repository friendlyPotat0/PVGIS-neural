extern crate neuroflow;

use std::fs;

use chrono::prelude::*;
use rand::Rng;

use neuroflow::data::DataSet;
use neuroflow::FeedForward;
use serde_json::Value;

pub struct Trainer {
    database_path: String,
    log_path: String,
}

impl Trainer {
    pub fn new(path: String) -> Trainer {
        Self {
            database_path: path,
            log_path: String::from(""),
        }
    }

    pub fn enable_error_logging(&mut self, log_path: String) {
        self.log_path = log_path;
    }

    pub fn train_neural_network(
        &mut self,
        ouput_path: &str,
        samples: usize,
        topology: &[i32],
        activation_function: neuroflow::activators::Type,
        eta: f64,
        alpha: f64,
    ) {
        // Get current UTC time
        let utc: DateTime<Utc> = Utc::now();
        let unix_timestamp = utc.timestamp();
        let unix_timestamp_str = unix_timestamp.to_string();
        // create neural network and specify topology
        let mut nn = FeedForward::new(topology);
        // define logfile for RMSE evolution
        if !self.log_path.is_empty() {
            let filename = format!(
                "{}/rmse_{}_samples_{}.txt",
                self.log_path, samples, unix_timestamp_str
            );
            nn.enable_error_logging(filename);
        }
        // define training data set
        let mut data: DataSet = DataSet::new();
        // read files from solar-radiation-database
        let database_dir = format!("{}", self.database_path);
        let files = fs::read_dir(database_dir).unwrap();
        // Iterate over files
        let mut i = 0;
        for file in files {
            if let Ok(file) = file {
                let file_path = file.path();
                if let Some(file_name) = file_path.file_name() {
                    if let Some(file_name_str) = file_name.to_str() {
                        i += 1;
                        println!("Loading [{i}]: {}!", file_name_str);
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
                                let unix_timestamp = Self::convert_to_unix_timestamp(timestamp_str);
                                // Normalize latitude and longitude (you need to implement normalization)
                                let latitude_norm = Self::normalize_latitude(latitude);
                                let longitude_norm = Self::normalize_longitude(longitude);
                                // Push data to training set
                                data.push(
                                    &[latitude_norm, longitude_norm, unix_timestamp as f64],
                                    &[power],
                                );
                            }
                        }
                        if i >= samples && samples != 0 {
                            break;
                        }
                    }
                }
            }
        }
        println!("Dataset filled!");
        // define neural network architecture
        nn.activation(activation_function)
            .learning_rate(eta)
            .momentum(alpha)
            .train(&data);
        println!("Training finished!");
        // save model to disk
        let filename = format!(
            "{}/model_{}_samples_{}.bin",
            ouput_path, samples, unix_timestamp_str
        );
        neuroflow::io::save(&mut nn, &filename).unwrap();
        println!("Model saved to disk!");
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
        let utc_datetime = Utc.from_utc_datetime(&datetime);
        // Convert to Unix timestamp
        let unix_timestamp = utc_datetime.timestamp();
        return unix_timestamp;
    }

    pub fn train_test() {
        let mut nn = FeedForward::new(&[2, 2, 1]);
        let logfile = String::from("resources/errors/rmse_00.txt");
        nn.enable_error_logging(logfile);
        let mut data: DataSet = DataSet::new();
        let mut i = -3.0;
        let mut rng = rand::thread_rng();
        while i <= 120.5 {
            let a: u8 = rng.gen_range(0..=1);
            let b: u8 = rng.gen_range(0..=1);
            data.push(&[a.into(), b.into()], &[(a ^ b).into()]);
            i += 0.05;
        }
        nn.activation(neuroflow::activators::Type::Tanh)
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
}
