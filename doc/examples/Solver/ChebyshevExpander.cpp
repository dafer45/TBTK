#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ChebyshevExpander");

//! [ChebyshevExpander]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

using namespace TBTK;

int main(){
	Initialize();

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
	Property::GreensFunction greensFunction
		= propertyExtractor.calculateGreensFunction({
			{{5, 5}, {5, 5}},
			{{2, 4}, {2, 4}}
		});
	for(unsigned int n = 0; n < greensFunction.getResolution(); n++)
		Streams::out << greensFunction({{5, 5}, {5, 5}}, n) << "\t";
	Streams::out << "\n";
	for(unsigned int n = 0; n < greensFunction.getResolution(); n++)
		Streams::out << greensFunction({{2, 4}, {2, 4}}, n) << "\t";
	Streams::out << "\n";
}
//! [ChebyshevExpander]
