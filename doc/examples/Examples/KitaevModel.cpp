#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("KitaevModel");

//! [KitaevModel]
#include "TBTK/Property/LDOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

complex<double> i(0, 1);

int main(){
	//Initialize TBTK.
	Initialize();

	//Parameters.
	const unsigned int SIZE = 400;
	const double t = -1;

	//Set up the Model.
	Model model;
	for(unsigned int x = 0; x < SIZE; x++){
		model << HoppingAmplitude(t, {x+1, 0}, {x, 0}) + HC;
		model << HoppingAmplitude(-t, {x+1, 1}, {x, 1}) + HC;
		model << HoppingAmplitude(0.1*i, {x+1, 1}, {x, 0}) + HC;
	}
	model.construct();

	//Set up the Solver.
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -2;
	const double UPPER_BOUND = 2;
	const unsigned int RESOLUTION = 1000;
	PropertyExtractor::Diagonalizer propertyExtractor;
	propertyExtractor.setSolver(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate the local density of states (LDOS) for all x, summing over
	//the Nambu space Subindex.
	Property::LDOS ldos
		= propertyExtractor.calculateLDOS({{_a_, IDX_SUM_ALL}});

	//Smooth the LDOS.
	const double SMOOTHING_SIGMA = 0.02;
	const unsigned int SMOOTHING_WINDOW = 51;
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the LDOS.
	Plotter plotter;
	plotter.plot({_a_, IDX_SUM_ALL}, ldos);
	plotter.save("figures/LDOS.png");

	//Calculate the wavefunctions.
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_, _a_}},
			{_a_}
		);

	//Plot the electron component of the Majorana wavefunction at E=0 (in
	//the middle of the energy spectrum).
	plotter.clear();
	plotter.plot({_a_, 0}, model.getBasisSize()/2, waveFunctions);
	plotter.save("figures/WaveFunction.png");
}
//! [KitaevModel]
