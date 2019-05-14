//////////////////////////////////////////////////// 
// Milestone 1: Neural Networks 28-line conversion
// File: main.cpp
// Date: July 16, 2018
// Author: Ricky Zaman
// Id: 121942171
// Email: rzaman6@myseneca.ca
///////////////////////////////////////////////////

// nine.cpp: Convert the neural net in nine lines of python to C++
// https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

// Embed the python code in the C++ code as a C++ comment to document what the C++ code is doing

// Why not grep the embedded python code, strip off the C++ comment, and run it?
//   grep //PY nine.cpp | sed 's=.*//PY ==' | grep -v '==' | python 

//PY from numpy import exp, array, random, dot
#include <algorithm>   // transform
#include <cmath>       // exp
#include <cstdlib>     // srand, rand
#include<functional>
#include <iomanip>     // setprecision
#include <iostream>
#include <numeric>     // inner_product
#include <vector>      // vectors vector<->, and matrices vector<vector<->>
using namespace std;

void vectorPrint(vector<double>& V) {
	for (auto col : V)
		cout << col << " ";
	cout << "\n";
}

void matrixPrint(vector<vector<double>>& M) {
	for (auto row : M)
		vectorPrint(row);
}

void matTranspose(vector<vector<double>>& Y, vector<vector<double>>& X) {
	size_t rows = X.size();    //  number of rows    for matrix X
	size_t cols = X[0].size(); //  number of columns for matrix X
	Y.resize(cols);             // set nunber of rows for Y
	for (auto&e : Y)             // set nunber of cols for each row of Y
		e.resize(rows);
	for (size_t r = 0; r < rows; r++)   // copy data
		for (size_t c = 0; c < cols; c++)
			Y[c][r] = X[r][c];
}

void matMult(vector<double>& Y, vector<vector<double>>& M, vector<double>& X) { // Y = M * X
	for (size_t i = 0; i < M.size(); i++) {
		Y[i] = inner_product(M[i].begin(), M[i].end(), X.begin(), 0.);
	}
}
	//PY class NeuralNetwork():
	class NeuralNetwork {
	public:
		vector<double> synaptic_weights;

		//PY def __init__(self) :
		//PY random.seed(1)
		//PY self.synaptic_weights = 2 * random.random((3, 1)) - 1

		NeuralNetwork(size_t s) {
			synaptic_weights.resize(s);
			srand(1);

			for (auto&e : synaptic_weights) {
				e = 2. * rand() / double(RAND_MAX) - 1.;
			}
		}

		//PY def __sigmoid(self, x) :
		//PY return 1 / (1 + exp(-x))

		double __sigmoid(double x) { return 1. / (1. + exp(-x)); }
		void __sigmoid(vector<double>& output, vector<double> input) {
			transform(input.begin(), input.end(), output.begin(), [this](double i) {return __sigmoid(i); });
											
		}


		//PY def __sigmoid_derivative(self, x) :
		//py return x * (1 - x)

		double __sigmoid_derivative(double x) { return x * (1. - x); }

		//PY def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations) :
		//PY	for iteration in xrange(number_of_training_iterations) :
		//PY		output = self.think(training_set_inputs)
		//PY		error = training_set_outputs - output
		//PY		adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
		//PY		self.synaptic_weights += adjustment

		void train(vector<vector<double>>& training_set_inputs, vector<double>& training_set_outputs, int number_of_training_iterations) {
			vector<double> output(training_set_outputs.size());
			vector<double> error(training_set_outputs.size());
			vector<vector<double>> training_set_inputsT;
			matTranspose(training_set_inputsT, training_set_inputs);
			vector<double>adjustment(training_set_inputs[0].size());

			for (int iteration = 0; iteration < number_of_training_iterations; iteration++) {

				think(output, training_set_inputs);

				transform(training_set_outputs.begin(), training_set_outputs.end(), 
					output.begin(), 
					error.begin(), 
					[](double t, double o) {return t - o; });

				transform(error.begin(), error.end(),
					output.begin(),
					error.begin(),
					[this](double e, double o) {return e*__sigmoid_derivative(o); });

				matMult(adjustment, training_set_inputsT, error);

				transform(synaptic_weights.begin(), synaptic_weights.end(),
					adjustment.begin(),
					synaptic_weights.begin(),
					[](double w, double a) {return w + a; });
				}
			}


		//PY def think(self, inputs) :
		//PY return self.__sigmoid(dot(inputs, self.synaptic_weights))
		void think(vector<double>& output, vector<vector<double>>& inputs) {
			matMult(output, inputs, synaptic_weights);
				__sigmoid(output, output);
		}

		double think(vector<double>& input) {
			return __sigmoid(inner_product(input.begin(), input.end(), synaptic_weights.begin(), 0.));
		}
	};

//PY if __name__ == "__main__":
	
int main() {

	//PY neural_network = NeuralNetwork()
	NeuralNetwork neural_network(3);

	//PY print "Random starting synaptic weights: "
	cout << "Random starting synaptic weights: ";

	//PY print neural_network.synaptic_weights
	vectorPrint(neural_network.synaptic_weights);

	//PY training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	//PY training_set_outputs = array([[0, 1, 1, 0]]).T
	vector<vector<double>> training_set_inputs = { { 0, 0, 1 },{ 1, 1, 1 },{ 1, 0, 1 },{ 0, 1, 1 } };
	vector<double> training_set_outputs = { 0,1,1,0 }; //1 row so its a vector, 1 row for each input element

	//PY neural_network.train(training_set_inputs, training_set_outputs, 10000)
	neural_network.train(training_set_inputs, training_set_outputs, 10000);

	//PY print "New synaptic weights after training: "
	cout << "New synaptic weights after training: ";

	//PY print neural_network.synaptic_weights
	vectorPrint(neural_network.synaptic_weights);

	//PY print "Considering new situation [1, 0, 0] -> ?: "
	cout << "Considering new situation [1,0,0] -> ?: ";

	//PY print neural_network.think(array([1, 0, 0]))
	vector<double> input = { 1,0,0 };
	cout << setprecision(8) << neural_network.think(input) << "\n"; //setprecision because c++ defaults 6 decimals but python does 8


	}

/*$python 28-main.py Output (Running through 1 time--Sample output):

Random starting synaptic weights :
[[-0.16595599]
[0.44064899]
[-0.99977125]]
New synaptic weights after training :
[[9.67299303]
[-0.2078435]
[-4.62963669]]
Considering new situation[1, 0, 0] -> ? :
[0.99993704]*/

/*
C++ Outut (Running through 1 time--Sample output):
Random starting synaptic weights : -0.997497 0.127171 -0.613392
New synaptic weights after training : 9.68252 -0.208184 -4.62919
Considering new situation[1, 0, 0] -> ? : 0.99993701

*/
