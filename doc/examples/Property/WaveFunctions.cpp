#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("WaveFunctions");

//! [WaveFunctions]
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/Property/WaveFunctions.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	Initialize();

#ifdef TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 41;
	const unsigned int SIZE_Y = 41;
#else //TBTK_DOCUMENTATION_NICE
	const unsigned int SIZE_X = 11;
	const unsigned int SIZE_Y = 11;
#endif //TBTK_DOCUMENTATION_NICE
	double t = 1;
	Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {0, -t});
	model.construct();

	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	PropertyExtractor::Diagonalizer propertyExtractor(solver);

	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_, _a_}},
			{_a_}
		);

#ifdef TBTK_DOCUMENTATION_NICE
	Streams::out << "waveFunctions({20, 20}, 100) = "
		<< waveFunctions({20, 20}, 100) << "\n";

	const unsigned int STATE = 4;

	Plotter plotter;
	plotter.plot({_a_, 20}, STATE, waveFunctions);
	plotter.save("figures/WaveFunction1D.png");

	plotter.clear();
	plotter.plot({_a_, _a_}, STATE, waveFunctions);
	plotter.save("figures/WaveFunction2D.png");
#else //TBTK_DOCUMENTATION_NICE
	Streams::out << "waveFunctions({5, 5}, 10) = "
		<< waveFunctions({5, 5}, 10) << "\n";

	const unsigned int STATE = 4;

	Plotter plotter;
	plotter.plot({_a_, 5}, STATE, waveFunctions);
	plotter.save("figures/WaveFunction1D.png");

	plotter.clear();
	plotter.plot({_a_, _a_}, STATE, waveFunctions);
	plotter.save("figures/WaveFunction2D.png");
#endif //TBTK_DOCUMENTATION_NICE
}
//! [WaveFunctions]
