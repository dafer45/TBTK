#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ArnoldiIterator");

//! [ArnoldiIterator]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	Model model;
	for(unsigned int x = 0; x < 200; x++)
		model << HoppingAmplitude(1, {x+1}, {x}) + HC;
	model.construct();

	Solver::ArnoldiIterator solver;
	solver.setModel(model);
	solver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(0.2);
	solver.setNumEigenValues(10);
	solver.setCalculateEigenVectors(true);
	solver.setNumLanczosVectors(20);
	solver.setMaxIterations(100);
	solver.run();

	const double LOWER_BOUND = -1;
	const double UPPER_BOUND = 1;
	const int RESOLUTION = 10;
	PropertyExtractor::ArnoldiIterator propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(LOWER_BOUND, UPPER_BOUND, RESOLUTION);

	Property::DOS dos = propertyExtractor.calculateDOS();
	Streams::out << dos << "\n";
}
//! [ArnoldiIterator]
