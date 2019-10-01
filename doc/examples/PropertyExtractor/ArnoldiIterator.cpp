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

	Streams::out << "----------------\n";

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

	Streams::out << "----------------\n";

	Property::DOS dos = propertyExtractor.calculateDOS();
	Streams::out << dos.getLowerBound() << "\n";
	Streams::out << dos.getUpperBound() << "\n";
	for(int n = 0; n < dos.getResolution(); n++)
		Streams::out << dos(n) << "\t";
	Streams::out << "\n";
}
//! [ArnoldiIterator]
