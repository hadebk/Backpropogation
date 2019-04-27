// data that will Neural Network training on it 
let training_data = [{
  inputs: [0, 0],
  targets: [0]
}, {
  inputs: [1, 0],
  targets: [1]
}, {
  inputs: [0, 1],
  targets: [1]
}, {
  inputs: [1, 1],
  targets: [0]
}];

// main function
function setup() {
  let nn = new NeuralNetwork(2, 2, 1); // (num_of_input,num_of_hidden,num_of_output)
  for(let i=0 ; i<50000 ; i++){
    let data = random(training_data);
    nn.train(data.inputs,data.targets);
  }
  
  // print output
  console.log(nn.feedforward([1,0]));
  console.log(nn.feedforward([0,1]));
  console.log(nn.feedforward([1,1]));
  console.log(nn.feedforward([0,0]));
}
