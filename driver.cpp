#include <string>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

#include "NeuralNetwork.h"

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;

int main(int argc, char* argv[]) {

	//RHMMUH005::NeuralNetwork neural1;

	// weights between input and hidden nodes
	MatrixXd weights_input_hidden(2, 2);

	weights_input_hidden(0, 0) = -1.0;
	weights_input_hidden(0, 1) = 0.0;

	weights_input_hidden(1, 0) = 0.0;
	weights_input_hidden(1, 1) = 1.0;

	// input values for the input nodes
	MatrixXd input(2, 1);

	input(0, 0) = 0.0;
	input(1, 0) = 1.0;

	// weights between hidden and output nodes
	MatrixXd weights_hidden_output(2, 2);

	weights_hidden_output(0, 0) = 1.0;
	weights_hidden_output(0, 1) = 0.0;

	weights_hidden_output(1, 0) = -1.0;
	weights_hidden_output(1, 1) = 1.0;

	MatrixXd targets(2, 1);

	targets(0, 0) = 1.0;
	targets(1, 0) = 0.0;

	RHMMUH005::NeuralNetwork neuralNetworkObj(input, weights_input_hidden, weights_hidden_output, targets);
	neuralNetworkObj.train();

	return 0;
}
