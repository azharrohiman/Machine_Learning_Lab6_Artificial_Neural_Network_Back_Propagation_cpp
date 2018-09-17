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

	NeuralNetwork::NeuralNetwork(MatrixXd input_values, MatrixXd input_hidden, MatrixXd hidden_output, MatrixXd target_values) {
		inputs = input_values;

		weights_input_hidden = input_hidden;
		weights_hidden_output = hidden_output;

		targets = target_values;
	}

	void NeuralNetwork::train() {
		//MatrixXd outputs = feedForward(inputs);

		ofstream output_file;
		output_file.open("output.txt");

		MatrixXd hidden_mat = weights_input_hidden * inputs;

		MatrixXd hidden(2, 1);
		hidden(0, 0) = sigmoid(hidden_mat(0, 0));
		hidden(1, 0) = sigmoid(hidden_mat(1, 0));

		output_file << "Hidden neuron 1: " << hidden(0, 0) << endl;
		output_file << "Hidden neuron 2: " << hidden(1, 0) << endl << endl;

		MatrixXd output_mat = weights_hidden_output * hidden;
		output_mat(0, 0) = sigmoid(output_mat(0, 0));
		output_mat(1, 0) = sigmoid(output_mat(1, 0));

		output_file << "Output neuron 1: " << output_mat(0, 0) << endl;
		output_file << "Output neuron 2: " << output_mat(1, 0) << endl << endl;

		output_file << "The MSE is: " << mse(output_mat(0, 0)) << endl;

		MatrixXd output_errors = targets - output_mat;

		//MatrixXd output_errors(2, 1);
		//output_errors(0, 0) = 0.5 * (targets(0, 0) - output_mat(0, 0)) * (targets(0, 0) - output_mat(0, 0));
		//output_errors(1, 0) = 0.5 * (targets(1, 0) - output_mat(1, 0)) * (targets(1, 0) - output_mat(1, 0));





		MatrixXd weights_hidden_output_transposed = weights_hidden_output.transpose();

		MatrixXd hidden_errors = weights_hidden_output_transposed * output_errors;

		cout << output_errors << endl;
	}

	MatrixXd NeuralNetwork::feedForward(MatrixXd input_values) {

		ofstream output_file;
		output_file.open("output.txt");

		MatrixXd hidden_mat = weights_input_hidden * input_values;

		MatrixXd hidden(2, 1);
		hidden(0, 0) = sigmoid(hidden_mat(0, 0));
		hidden(1, 0) = sigmoid(hidden_mat(1, 0));

		output_file << "Hidden neuron 1: " << hidden(0, 0) << endl;
		output_file << "Hidden neuron 2: " << hidden(1, 0) << endl << endl;

		MatrixXd output_mat = weights_hidden_output * hidden;
		output_mat(0, 0) = sigmoid(output_mat(0, 0));
		output_mat(1, 0) = sigmoid(output_mat(1, 0));

		output_file << "Output neuron 1: " << output_mat(0, 0) << endl;
		output_file << "Output neuron 2: " << output_mat(1, 0) << endl << endl;

		output_file << "The MSE is: " << mse(output_mat(0, 0)) << endl;

		return output_mat;
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
