#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Diagonalizer");

//! [Diagonalizer]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	Model model;
	model << HoppingAmplitude(1, {0}, {1});
	model << HoppingAmplitude(1, {1}, {0});
	model.construct();

	Solver::Diagonalizer solver;
	solver.setVerbose(true);
	solver.setModel(model);
	solver.run();

	Streams::out << "--- Results ----\n";

	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	for(int n = 0; n < solver.getModel().getBasisSize(); n++){
		Streams::out
			<< propertyExtractor.getEigenValue(n) << "\t["
			<< propertyExtractor.getAmplitude(n, {0}) << "\t"
			<< propertyExtractor.getAmplitude(n, {1}) << "]\n";
	}
}
//! [Diagonalizer]
