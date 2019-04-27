//(1)will adjust all your data points between 0 and 1, 
//If you want to adjust it to 2 or a higher value, just change the numerator and you are good to go.
//(2)to scale the data in some given specific range with a threshold.
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x)); 
}

function dsigmoid(y) {
  return y * (1 - y);
}
//-----------------------------------------------------------------
class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;
    // generate the weights
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes); // generate matrix for weights
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes); // generate matrix for weights
    this.weights_ih.randomize(); // put value of weights in matrix(weights_ih)
    this.weights_ho.randomize(); // put value of weights in matrix(weights_ho)
    
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();
    this.learning_rate = 0.1;
  }

  feedforward(input_array) {//calc values of nodes in hidden layer and output layer
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array); // convert array to matrix
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(sigmoid); // Apply a sigmoid function to every element of matrix

    // Generating the output's output!
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid); // Apply a sigmoid function to every element of matrix

    // convert againe to array
    return output.toArray();
  }

  train(input_array, target_array) {
    //------------------------------------------------------
    // feedforward() work..
      
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(sigmoid);

    // Generating the output's output!
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);
    //-------------------------------------------------------
    
    //-----------------------------------------------------------------------------------------
    // calc error output and update ther weights
      
    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);

    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = Matrix.map(outputs, dsigmoid); // execute => outputs * (1 - outputs)
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    // Calculate deltas of weights
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);
    
    // update the weights by deltas
    this.weights_ho.add(weight_ho_deltas);
    // update the bias by its deltas (which is just the gradients)
    this.bias_o.add(gradients);
    //-----------------------------------------------------------------------------------------

    //-----------------------------------------------------------------------------------------
    // calc error hidden and update ther weights
      
    // Calculate the hidden layer errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors); // execute => outputs * (1 - outputs)

    // Calculate hidden gradient07
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    // Calcuate input->hidden deltas
let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
    
    // update the weights by deltas
    this.weights_ih.add(weight_ih_deltas);
    // update the bias by its deltas (which is just the gradients)
    this.bias_h.add(hidden_gradient);

    //-----------------------------------------------------------------------------------------

  }

}
