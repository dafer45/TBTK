#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Magnetization");

//! [Magnetization]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	const unsigned int SIZE_X = 11;
	const unsigned int SIZE_Y = 11;
	double t = 1;
	double J = 1;
	Model model
		= Models::SquareLattice({SIZE_X, SIZE_Y, IDX_SPIN}, {0, t});
	model << HoppingAmplitude(
		J,
		{SIZE_X/2, SIZE_Y/2, 0},
		{SIZE_X/2, SIZE_Y/2, 0}
	);
	model << HoppingAmplitude(
		-J,
		{SIZE_X/2, SIZE_Y/2, 1},
		{SIZE_X/2, SIZE_Y/2, 1}
	);
	model.setChemicalPotential(-2);
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{_a_, _a_, IDX_SPIN}
		});

	Streams::out << "magnetization({5, 5, IDX_SPIN}) = "
		<< magnetization({5, 5, IDX_SPIN}) << "\n";

	Plotter plotter;
	plotter.setTitle("Magnetization projected on the z-axis");
	plotter.plot({_a_, _a_, IDX_SPIN}, {0, 0, 1}, magnetization);
	plotter.save("figures/MagnetizationZ.png");
}
//! [Magnetization]
