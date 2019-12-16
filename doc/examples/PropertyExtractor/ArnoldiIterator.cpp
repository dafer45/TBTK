#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ArnoldiIterator");

//! [ArnoldiIterator]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

#include <complex>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

complex<double> i(0, 1);

int main(){
	Initialize();

	Model model;
	for(unsigned int x = 0; x < 400; x++){
		model << HoppingAmplitude(-1, {x+1, 0}, {x, 0}) + HC;
		model << HoppingAmplitude(1, {x+1, 1}, {x, 1}) + HC;
		model << HoppingAmplitude(0.1*i, {x+1, 1}, {x, 0}) + HC;
	}
	model.construct();

	Solver::ArnoldiIterator solver;
	solver.setModel(model);
	solver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(0.01);
	solver.setNumEigenValues(200);
	solver.setCalculateEigenVectors(true);
	solver.setNumLanczosVectors(400);
	solver.setMaxIterations(500);
	solver.run();

	const double LOWER_BOUND = -1;
	const double UPPER_BOUND = 1;
	const int RESOLUTION = 200;
	PropertyExtractor::ArnoldiIterator propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Plotter plotter;
	const double SMOOTHING_SIGMA = 0.02;
	const unsigned int SMOOTHING_WINDOW = 51;

	Property::DOS dos = propertyExtractor.calculateDOS();
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	Streams::out << dos << "\n";
	plotter.plot(dos);
	plotter.save("figures/DOS.png");

	Property::LDOS ldos
		= propertyExtractor.calculateLDOS({{_a_, IDX_SUM_ALL}});
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	Streams::out << ldos << "\n";
	plotter.clear();
	plotter.plot({_a_, IDX_SUM_ALL}, ldos);
	plotter.save("figures/LDOS.png");
}
//! [ArnoldiIterator]
