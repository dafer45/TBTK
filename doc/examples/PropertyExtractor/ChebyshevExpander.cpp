#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ChebyshevExpander");

//! [ChebyshevExpander]
#include "TBTK/Model.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	const unsigned int SIZE_X = 100;
	const unsigned int SIZE_Y = 100;
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.construct();

	Solver::ChebyshevExpander solver;
	solver.setModel(model);
	solver.setScaleFactor(10);
	solver.setCalculateCoefficientsOnGPU(false);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(true);
	solver.setNumCoefficients(200);

	const double LOWER_BOUND = -5;
	const double UPPER_BOUND = 5;
	const int RESOLUTION = 200;
	PropertyExtractor::ChebyshevExpander propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::LDOS ldos = propertyExtractor.calculateLDOS({{50, 50}});
	Streams::out << ldos << "\n";

	Plotter plotter;
	plotter.plot({50, 50}, ldos);
	plotter.save("figures/LDOS.png");
}
//! [ChebyshevExpander]
