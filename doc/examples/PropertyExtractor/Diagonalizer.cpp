#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Diagonalizer");

//! [Diagonalizer]
#include "TBTK/Model.h"
#include "TBTK/Models/SquareLattice.h"
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
	const int SIZE_X = 40;
	const int SIZE_Y = 40;
#else //TBTK_DOCUMENTATION_NICE
	const int SIZE_X = 10;
	const int SIZE_Y = 10;
#endif //TBTK_DOCUMENTATION_NICE
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, t});
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const int RESOLUTION = 200;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Plotter plotter;
	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 51;

	Property::DOS dos = propertyExtractor.calculateDOS();
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	Streams::out << dos << "\n";
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, 0},	//All points of the form {x, 0}
#ifdef TBTK_DOCUMENTATION_NICE
		{20, _a_},	//All points of the form {20, y}
		{10, 10},
		{10, 11},
		{11, 10},
		{11, 11}
#else //TBTK_DOCUMENTATION_NICE
		{5, _a_},	//All points of the form {5, y}
		{2, 2},
		{2, 3},
		{3, 2},
		{3, 3}
#endif //TBTK_DOCUMENTATION_NICE
	});
	Streams::out << density << "\n";
	plotter.clear();
	plotter.plot({_a_, _a_}, density);
	plotter.save("figures/Density.png");

#ifdef TBTK_DOCUMENTATION_NICE
	Property::LDOS ldos = propertyExtractor.calculateLDOS({{_a_, 20}});
#else //TBTK_DOCUMENTATION_NICE
	Property::LDOS ldos = propertyExtractor.calculateLDOS({{_a_, 5}});
#endif //TBTK_DOCUMENTATION_NICE
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	Streams::out << ldos << "\n";
	plotter.clear();
#ifdef TBTK_DOCUMENTATION_NICE
	plotter.plot({_a_, 20}, ldos);
#else //TBTK_DOCUMENTATION_NICE
	plotter.plot({_a_, 5}, ldos);
#endif //TBTK_DOCUMENTATION_NICE
	plotter.save("figures/LDOS.png");
}
//! [Diagonalizer]
