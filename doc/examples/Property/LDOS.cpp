#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("LDOS");

//! [LDOS]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	const double LOWER_BOUND = -5;
	const double UPPER_BOUND = 5;
	const unsigned int RESOLUTION = 100;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::LDOS ldos = propertyExtractor.calculateLDOS({{_a_, _a_}});

	Streams::out << ldos << "\n";
	Streams::out << "ldos({5, 5}, 50) = " << ldos({5, 5}, 50) << "\n";

	double integratedLDOS = 0;
	double dE = ldos.getDeltaE();
	for(unsigned int n = 0; n < ldos.getResolution(); n++)
		integratedLDOS += ldos({5, 5}, n)*dE;
	Streams::out << "Integrated LDOS: " << integratedLDOS << "\n";
}
//! [LDOS]
