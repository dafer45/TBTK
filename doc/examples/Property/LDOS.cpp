#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("LDOS");

//! [LDOS]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	Initialize();

#ifdef TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 40;
	const unsigned int SIZE_Y = 40;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 10;
	const unsigned int SIZE_Y = 10;
#endif //TBTK_DOCUMENTATION_NICE
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

	Property::LDOS ldos = propertyExtractor.calculateLDOS({{_a_, _a_}});

	Streams::out << ldos << "\n";
#ifdef TBTK_DOCUMENTATION_NICE
	Streams::out << "ldos({20, 20}, 50) = " << ldos({20, 20}, 50) << "\n";
#else //TBTK_DOCUMENTATION_NICE
	Streams::out << "ldos({5, 5}, 50) = " << ldos({5, 5}, 50) << "\n";
#endif //TBTK_DOCUMENTATION_NICE

	double integratedLDOS = 0;
	double dE = ldos.getDeltaE();
	for(unsigned int n = 0; n < ldos.getResolution(); n++)
#ifdef TBTK_DOCUMENTATION_NICE
		integratedLDOS += ldos({20, 20}, n)*dE;
#else //TBTK_DOCUMENTATION_NICE
		integratedLDOS += ldos({5, 5}, n)*dE;
#endif //TBTK_DOCUMENTATION_NICE
	Streams::out << "Integrated LDOS: " << integratedLDOS << "\n";

	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 51;
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	Plotter plotter;
#ifdef TBTK_DOCUMENTATION_NICE
	plotter.plot({_a_, 20}, ldos);
#else //TBTK_DOCUMENTATION_NICE
	plotter.plot({_a_, 5}, ldos);
#endif //TBTK_DOCUMENTATION_NICE
	plotter.save("figures/LDOS0.png");

	plotter.clear();
#ifdef TBTK_DOCUMENTATION_NICE
	plotter.plot({20, 20}, ldos);
#else //TBTK_DOCUMENTATION_NICE
	plotter.plot({5, 5}, ldos);
#endif //TBTK_DOCUMENTATION_NICE
	plotter.save("figures/LDOS1.png");
}
//! [LDOS]
