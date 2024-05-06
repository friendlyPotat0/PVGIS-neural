extern crate neuroflow;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;
use rand::Rng;


fn main(){
    /*
        Define a neural network with 1 neuron in input layers. The network contains 4 hidden layers.
        And, such as our function returns a single value, it is reasonable to have 1 neuron in the output layer.
    */
    let mut nn = FeedForward::new(&[3, 64, 32, 1]);
    
    let logfile = String::from("resources/rmse_02.txt");
    nn.set_logfile(&logfile);
    
    let mut data: DataSet = DataSet::new();
    // data.push(&[a.into(), b.into()], &[(a ^ b).into()]);
    
    // Here, we set the necessary parameters and train the neural network by our DataSet with 50 000 iterations
    nn.activation(Tanh)
        .learning_rate(0.1)
        .momentum(0.15)
        .train(&data, 2500);

    // let mut res;
    
    // Let's check the result
    /* i = 0.0;
    while i <= 0.3{
        let a: u8 = rng.gen_range(0..=1);
        let b: u8 = rng.gen_range(0..=1);
        res = nn.calc(&[a.into(), b.into()])[0];
        println!("for [{:.3}], [{:.3}] = [{:.3}] -> [{:.3}]", a, b, a ^ b, res);
        i += 0.07;
    } */
}
