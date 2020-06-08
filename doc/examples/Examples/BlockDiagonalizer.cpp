#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("BlockDiagonalizer");

//! [BlockDiagonalizer]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/Range.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>
#include <cmath>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

int main(){
	//Initialize TBTK.
	Initialize();

	//Parameters.
	const int NUM_K_POINTS = 10000;
	double a = 1;

	//Set up the Model.
	Model model;
	Range K(0, 2*M_PI, NUM_K_POINTS);
	for(int k = 0; k < NUM_K_POINTS; k++)
		model << HoppingAmplitude(cos(K[k]*a), {k, 0}, {k, 1}) + HC;
	model.setChemicalPotential(-0.5);
	model.setTemperature(300);
	model.construct();

	//Set up the Solver.
	Solver::BlockDiagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -2;
	const double UPPER_BOUND = 2;
	const int RESOLUTION = 200;
	PropertyExtractor::BlockDiagonalizer propertyExtractor;
	propertyExtractor.setSolver(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate the density of states (DOS).
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Smooth the DOS
	const double SMOOTHING_SIGMA = 0.01;
	const unsigned int SMOOTHING_WINDOW = 51;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);

	//Plot the DOS.
	Plotter plotter;
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	//Caluclate the density (momentum space density).
	Property::Density density = propertyExtractor.calculateDensity({
		//All k-points, summing over the second subindex.
		{_a_, IDX_SUM_ALL}
	});

	//Plot the Density.
	plotter.clear();
	plotter.setAxes({{0, {0, 2*M_PI}}});
	plotter.plot({_a_, IDX_SUM_ALL}, density);
	plotter.setLabelX("k");
	plotter.save("figures/Density.png");
}
//! [BlockDiagonalizer]
