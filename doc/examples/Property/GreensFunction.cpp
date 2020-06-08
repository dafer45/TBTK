#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("GreensFunction");

//! [GreensFunction]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

using namespace TBTK;

int main(){
	Initialize();

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
	PropertyExtractor::Diagonalizer propertyExtractor;
	propertyExtractor.setSolver(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::GreensFunction greensFunction
		= propertyExtractor.calculateGreensFunction({
			{{5, 5}, {5, 5}}
		});

	Streams::out << greensFunction << "\n";
	Streams::out << "greensFunction({{5, 5}, {5, 5}}, 50) = "
		<< greensFunction({{5, 5}, {5, 5}}, 50) << "\n";
}
//! [GreensFunction]
