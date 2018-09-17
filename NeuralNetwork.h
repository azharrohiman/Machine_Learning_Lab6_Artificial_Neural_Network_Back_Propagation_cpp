#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <memory>
#include <string>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace Eigen;
using Eigen::MatrixXd;

namespace RHMMUH005 {
	
	class NeuralNetwork {

		private:
			MatrixXd inputs;

			MatrixXd weights_input_hidden;
			MatrixXd weights_hidden_output;

		public:
			NeuralNetwork();
			NeuralNetwork(MatrixXd, MatrixXd, MatrixXd);

			void feedForward();

			double sigmoid(double);

			double mse(double);

	};
}

#endif
