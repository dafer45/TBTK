#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("SpinPolarizedLDOS");

//! [SpinPolarizedLDOS]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
#ifdef TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 21;
	const unsigned int SIZE_Y = 21;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 11;
	const unsigned int SIZE_Y = 11;
#endif //TBTK_DOCUMENTATION_NICE
	double t = 1;
	double J = 1;
	Model model = Models::SquareLattice(
		{SIZE_X, SIZE_Y, IDX_SPIN},
		{0, t}
	);
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
#ifdef TBTK_DOCUMENTATION_NICE
	Streams::out << spinPolarizedLDOS({10, 10, IDX_SPIN}, 49) << "\n";
#else //TBTK_DOCUMENTATION_NICE
	Streams::out << spinPolarizedLDOS({5, 5, IDX_SPIN}, 49) << "\n";
#endif //TBTK_DOCUMENTATION_NICE

	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 51;
	spinPolarizedLDOS = Smooth::gaussian(
		spinPolarizedLDOS,
		SMOOTHING_SIGMA,
		SMOOTHING_WINDOW
	);

	Plotter plotter;
	plotter.setNumContours(100);
#ifdef TBTK_DOCUMENTATION_NICE
	plotter.plot({_a_, 10, IDX_SPIN}, {0, 0, 1}, spinPolarizedLDOS);
#else //TBTK_DOCUMENTATION_NICE
	plotter.plot({_a_, 5, IDX_SPIN}, {0, 0, 1}, spinPolarizedLDOS);
#endif
	plotter.save("figures/SpinPolarizedLDOS.png");
}
//! [SpinPolarizedLDOS]
