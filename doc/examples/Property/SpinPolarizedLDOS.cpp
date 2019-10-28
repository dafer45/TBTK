#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("SpinPolarizedLDOS");

//! [SpinPolarizedLDOS]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	double t = 1;
	Model model = Models::SquareLattice(
		{SIZE_X, SIZE_Y, IDX_SPIN},
		{0, t}
	);
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

	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{_a_, _a_, IDX_SPIN}
		});

	Streams::out << spinPolarizedLDOS << "\n";
	Streams::out << spinPolarizedLDOS({5, 5, IDX_SPIN}, 49) << "\n";
}
//! [SpinPolarizedLDOS]
