#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("BlockDiagonalizer");

//! [BlockDiagonalizer]
#include "TBTK/Array.h"
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

#include <complex>

using namespace std;
using namespace TBTK;

int main(){
	Initialize();

	Model model;
	//Block 0.
	model << HoppingAmplitude(1, {0, 0}, {0, 1});
	model << HoppingAmplitude(1, {0, 1}, {0, 0});
	//Block 1.
	model << HoppingAmplitude(2, {1, 0}, {1, 1});
	model << HoppingAmplitude(2, {1, 1}, {1, 0});
	model.construct();

	Solver::BlockDiagonalizer solver;
	solver.setVerbose(true);
	solver.setModel(model);
	solver.run();

	PropertyExtractor::BlockDiagonalizer propertyExtractor(solver);

	Streams::out << "----------------\n";

	//Blockwise access.
	Array<double> eigenValues({2, 2});
	Array<complex<double>> eigenVectors({4, 2});
	for(unsigned int block = 0; block < 2; block++){
		for(unsigned int state = 0; state < 2; state++){
			eigenValues[{block, state}]
				= propertyExtractor.getEigenValue(
					{block},
					state
				);
			for(unsigned int n = 0; n < 2; n++){
				eigenVectors[{2*block + state, n}]
					= propertyExtractor.getAmplitude(
						{block},
						state,
						{n}
					);
			}
		}
	}
	Streams::out << eigenValues << "\n";
	Streams::out << eigenVectors << "\n";

	Streams::out << "----------------\n";

	//Global access.
	eigenValues = Array<double>({4});
	eigenVectors = Array<complex<double>>({4, 4});
	for(
		unsigned int state = 0;
		(int)state < solver.getModel().getBasisSize();
		state++
	){
		eigenValues[{state}] = propertyExtractor.getEigenValue(state);
		for(unsigned int block = 0; block < 2; block++){
			for(unsigned int n = 0; n < 2; n++){
				eigenVectors[{state, 2*block + n}]
					= propertyExtractor.getAmplitude(
						state,
						{block, n}
					);
			}
		}
	}
	Streams::out << eigenValues << "\n";
	Streams::out << eigenVectors << "\n";
}
//! [BlockDiagonalizer]
