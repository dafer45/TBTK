#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Density");

//! [Density]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/Density.h"
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

	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	Property::Density density
		= propertyExtractor.calculateDensity({{_a_, _a_}});

	Streams::out << "density({5, 5}) = " << density({5, 5}) << "\n";
}
//! [Density]
