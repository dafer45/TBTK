#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ArnoldiIterator");

//! [ArnoldiIterator]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

using namespace TBTK;

int main(){
	Initialize();

	Model model;
	for(unsigned int x = 0; x < 200; x++)
		model << HoppingAmplitude(1, {x+1}, {x}) + HC;
	model.construct();

	Solver::ArnoldiIterator solver;
	solver.setVerbose(true);
	solver.setModel(model);
	solver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(1.0);
	solver.setNumEigenValues(10);
	solver.setCalculateEigenVectors(true);
	solver.setNumLanczosVectors(20);
	solver.setMaxIterations(100);
	solver.run();

	Streams::out << "----------------\n";

	//Print the eigenvalues and the amplitude on site 50.
	PropertyExtractor::ArnoldiIterator propertyExtractor(solver);
	for(int n = 0; n < 10; n++){
		Streams::out
			<< propertyExtractor.getEigenValue(n) << "\t"
			<< propertyExtractor.getAmplitude(n, {50}) << "\n";
	}
}
//! [ArnoldiIterator]
