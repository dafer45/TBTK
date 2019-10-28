#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Magnetization");

//! [Magnetization]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	const double t = 1;
	const bool INCLUDE_SPIN_INDEX = true;
	Model model = Models::SquareLattice(
		{SIZE_X, SIZE_Y},
		{0, t},
		INCLUDE_SPIN_INDEX
	);
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{_a_, _a_, IDX_SPIN}
		});

	Streams::out << magnetization({5, 5, IDX_SPIN}) << "\n";
}
//! [Magnetization]
