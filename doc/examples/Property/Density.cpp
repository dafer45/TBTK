#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Density");

//! [Density]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/Density.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.setChemicalPotential(-2);
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	Property::Density density
		= propertyExtractor.calculateDensity({{_a_, _a_}});

	Streams::out << "density({5, 5}) = " << density({5, 5}) << "\n";

	Plotter plotter;
	plotter.plot({_a_, _a_}, density);
	plotter.save("figures/Density.png");

	plotter.clear();
	plotter.plot({_a_, 5}, density);
	plotter.save("figures/DensityCut.png");
}
//! [Density]
