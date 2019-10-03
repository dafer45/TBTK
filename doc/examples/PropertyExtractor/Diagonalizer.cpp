#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Diagonalizer");

//! [Diagonalizer]
#include "TBTK/Model.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	const int SIZE_X = 10;
	const int SIZE_Y = 10;
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const int RESOLUTION = 10;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::DOS dos = propertyExtractor.calculateDOS();
	Streams::out << dos << "\n";

	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, 0},	//All points of the form {x, 0}
		{5, _a_}	//All points of the form {5, y}
	});
	Streams::out << density << "\n";
}
//! [Diagonalizer]
