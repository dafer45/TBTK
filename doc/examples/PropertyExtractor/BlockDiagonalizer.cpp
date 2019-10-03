#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("BlockDiagonalizer");

//! [BlockDiagonalizer]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Streams.h"

#include <complex>
#include <cmath>

using namespace std;
using namespace TBTK;

int main(){
	const int NUM_K_POINTS = 10;

	Model model;
	for(int k = 0; k < NUM_K_POINTS; k++){
		model << HoppingAmplitude(
			cos(2*M_PI*k/10.),
			{k, 0},
			{k, 1}
		) + HC;
	}
	model.setChemicalPotential(-0.5);
	model.construct();

	Solver::BlockDiagonalizer solver;
	solver.setModel(model);
	solver.run();

	const double LOWER_BOUND = -1;
	const double UPPER_BOUND = 1;
	const int RESOLUTION = 10;
	PropertyExtractor::BlockDiagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::DOS dos = propertyExtractor.calculateDOS();
	Streams::out << dos << "\n";

	Property::Density density = propertyExtractor.calculateDensity({
		//All k-points, summing over the second subindex.
		{_a_, IDX_SUM_ALL}
	});
	Streams::out << density << "\n";
}
//! [BlockDiagonalizer]
