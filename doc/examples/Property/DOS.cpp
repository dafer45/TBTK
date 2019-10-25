#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("DOS");

//! [DOS]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/DOS.h"
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

	Property::DOS dos = propertyExtractor.calculateDOS();

	Streams::out << dos << "\n";
	Streams::out << "dos(50) = " << dos(50) << "\n";

	double integratedDOS = 0;
	double dE = dos.getDeltaE();
	for(unsigned int n = 0; n < dos.getResolution(); n++)
		integratedDOS += dos(n)*dE;
	Streams::out << "Integrated DOS: " << integratedDOS << "\n";
}
//! [DOS]
