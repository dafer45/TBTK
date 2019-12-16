#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("EigenValues");

//! [EigenValues]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

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

	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	Property::EigenValues eigenValues
		= propertyExtractor.getEigenValues();

	Streams::out << "eigenValues(0) = " << eigenValues(0) << "\n";
	Streams::out << "eigenValues(10) = " << eigenValues(10) << "\n";
	Streams::out << "eigenValues(99) = " << eigenValues(99) << "\n";

	Plotter plotter;
	plotter.plot(eigenValues);
	plotter.save("figures/EigenValues.png");
}
//! [EigenValues]
