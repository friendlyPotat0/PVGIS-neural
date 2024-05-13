use std::io::Write;
extern crate neuroflow;

use std::fs::{self, OpenOptions};

use chrono::prelude::*;
use neuroflow::FeedForward;
use serde_json::Value;

pub struct Trainer {
    database: String,
    log_path: String,
    epoch_interval: i32,
}

impl Trainer {
    pub fn new(path: String) -> Trainer {
        Self {
            database: path,
            log_path: String::from(""),
            epoch_interval: 0,
        }
    }

    pub fn enable_error_logging(&mut self, log_path: String, epoch_interval: i32) {
        self.log_path = log_path;
        self.epoch_interval = epoch_interval;
    }

    pub fn train_neural_network(
        &mut self,
        ouput_path: &str,
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
        let mut log_filename: String = String::from("");
        if !self.log_path.is_empty() {
            log_filename = format!("{}/rmse_{}.txt", self.log_path, unix_timestamp_str);
            nn.enable_error_logging(log_filename.clone(), self.epoch_interval);
        }
        // define training data set
        // let data = DataSet::from_csv(&self.database).unwrap();
        // println!("Dataset filled!");
        // define neural network architecture
        nn.activation(activation_function)
            .learning_rate(eta)
            .momentum(alpha)
            .train_mem_efficient(&self.database);
        println!("Training finished!\n");
        // save model to disk
        let model_filename = format!("{}/model_{}.bin", ouput_path, unix_timestamp_str);
        neuroflow::io::save(&mut nn, &model_filename).unwrap();
        println!("Trained model saved to disk at {}", model_filename);
        if !self.log_path.is_empty() {
            println!("Log file saved at: {}", log_filename);
        }
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

    pub fn synthesize_pvgis_data(&self, database_path: &str, output_path: &str) {
        // read files from solar-radiation-database
        let files = fs::read_dir(database_path).unwrap();
        let file_count = fs::read_dir(database_path).unwrap().count();
        // generate output file
        let filename = format!("{output_path}/PVGIS_dataset_{file_count}.csv");
        let mut output_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .unwrap();
        // Iterate over files
        let mut i = 0;
        for file in files {
            if let Ok(file) = file {
                let file_path = file.path();
                if let Some(file_name) = file_path.file_name() {
                    if let Some(file_name_str) = file_name.to_str() {
                        i += 1;
                        println!("Processing [{i}]: {}!", file_name_str);
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
                                let line = format!(
                                    "{latitude_norm},{longitude_norm},{unix_timestamp},-,{power}"
                                );
                                writeln!(output_file, "{}", line).unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
}
