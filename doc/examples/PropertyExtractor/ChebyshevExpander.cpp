#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ChebyshevExpander");

//! [ChebyshevExpander]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	Model model;
	for(int x = 0; x < 10; x++){
		for(int y = 0; y < 10; y++){
			model << HoppingAmplitude(1, {x+1, y}, {x, y}) + HC;
			model << HoppingAmplitude(1, {x, y+1}, {x, y}) + HC;
		}
	}
	model.construct();

	Solver::ChebyshevExpander solver;
	solver.setModel(model);
	solver.setScaleFactor(10);
	solver.setCalculateCoefficientsOnGPU(false);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(true);
	solver.setNumCoefficients(10);

	const double LOWER_BOUND = -1;
	const double UPPER_BOUND = 1;
	const int RESOLUTION = 10;
	PropertyExtractor::ChebyshevExpander propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::LDOS ldos = propertyExtractor.calculateLDOS({{5, 5}});
	Streams::out << ldos << "\n";
}
//! [ChebyshevExpander]
