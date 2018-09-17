#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <math.h>

#include "NeuralNetwork.h"

using namespace std;

namespace RHMMUH005 
{
	NeuralNetwork::NeuralNetwork() {

	}

	NeuralNetwork::NeuralNetwork(MatrixXd input_values, MatrixXd input_hidden, MatrixXd hidden_output) {
		inputs = input_values;

		weights_input_hidden = input_hidden;
		weights_hidden_output = hidden_output;
	}

	void NeuralNetwork::train(MatrixXd inputs, MatrixXd targets) {

	}

	void NeuralNetwork::feedForward() {

		ofstream output_file;
		output_file.open("output.txt");

		MatrixXd hidden_mat = weights_input_hidden * inputs;

		MatrixXd hidden(2, 1);
		hidden(0, 0) = sigmoid(hidden_mat(0, 0));
		hidden(1, 0) = sigmoid(hidden_mat(1, 0));

		output_file << "First neuron: " << hidden(0, 0) << endl;
		output_file << "Second neuron: " << hidden(1, 0) << endl << endl;

		MatrixXd output_mat = weights_hidden_output * hidden;

		output_file << "Output neuron: " << sigmoid(output_mat(0, 0)) << endl << endl;

		output_file << "The MSE is: " << mse(sigmoid(output_mat(0, 0))) << endl;
	}

	double NeuralNetwork::sigmoid(double val) {
		double exp_value;
     		double return_value;

     		/*** Exponential calculation ***/
     		exp_value = exp((double) -val);

     		/*** Final sigmoid value ***/
     		return_value = 1 / (1 + exp_value);

     		return return_value;
	}

	double NeuralNetwork::mse(double val) {
		return 0.5 * pow((0.36 - val), 2);
	}
}
