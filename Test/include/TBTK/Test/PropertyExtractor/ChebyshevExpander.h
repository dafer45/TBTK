#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/Solver/Diagonalizer.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();
const double EPSILON_10000 = 10000*std::numeric_limits<double>::epsilon();
const double CHEMICAL_POTENTIAL = 1;

//TODO
//Add tests for most functions.

TEST(ChebyshevExpander, calculateLDOS0){
	const unsigned int SIZE = 10;
	Model model;
	for(unsigned int n = 0; n < SIZE; n++)
		model << HoppingAmplitude(-1, {n}, {(n+1)%SIZE}) + HC;
	model.construct();

	Solver::ChebyshevExpander solver;
	solver.setModel(model);
	const double SCALE_FACTOR = 10;
	solver.setScaleFactor(SCALE_FACTOR);

	ChebyshevExpander propertyExtractor(solver);
	double fractions[2] = {0.9, 0.7};
	for(unsigned int n = 0; n < 2; n++){
		const double LOWER_BOUND = -SCALE_FACTOR*fractions[n];
		const double UPPER_BOUND = SCALE_FACTOR*fractions[n];
		const unsigned int RESOLUTION = 1000;
		propertyExtractor.setEnergyWindow(LOWER_BOUND, UPPER_BOUND, RESOLUTION);
		Property::LDOS ldos = propertyExtractor.calculateLDOS({{0}});
		double dE = ldos.getDeltaE();
		double integratedLDOS = 0;
		for(unsigned int n = 0; n < ldos.getResolution(); n++)
			integratedLDOS += ldos({0}, n)*dE;
		EXPECT_NEAR(integratedLDOS, 1, 0.001);
	}
}

TEST(ChebyshevExpander, calculateLDOS1){
	const unsigned int SIZE = 10;
	Model model;
	for(unsigned int n = 0; n < SIZE; n++)
		model << HoppingAmplitude(-1, {n}, {(n+1)%SIZE}) + HC;
	model.construct();

	Solver::ChebyshevExpander solver;
	solver.setModel(model);
	const double SCALE_FACTOR = 10;
	solver.setScaleFactor(SCALE_FACTOR);

	Solver::Diagonalizer solverReference;
	solverReference.setModel(model);
	solverReference.run();

	ChebyshevExpander propertyExtractor(solver);
	Diagonalizer propertyExtractorReference;
	propertyExtractorReference.setSolver(solverReference);

	const double LOWER_BOUND = -SCALE_FACTOR*0.9;
	const double UPPER_BOUND = SCALE_FACTOR*0.9;
	const unsigned int RESOLUTION = 1000;
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	propertyExtractorReference.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Property::LDOS ldos = propertyExtractor.calculateLDOS({{0}});
	Property::LDOS ldosReference
		= propertyExtractorReference.calculateLDOS({{0}});

	//Smooth the result to even out differences. The Diagonalizer
	//genereates very unsmooth data, while the ChebyshevExpander generates
	//smoother data. Putting them both through a smoother evens out the
	//differences.
	const double SMOOTHING_SIGMA = 0.2;
	const unsigned int SMOOTHING_WINDOW = 101;
	ldos = Smooth::gaussian(ldos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	ldosReference = Smooth::gaussian(
		ldosReference,
		SMOOTHING_SIGMA,
		SMOOTHING_WINDOW
	);

	for(unsigned int n = 0; n < ldos.getResolution(); n++)
		EXPECT_NEAR(ldos({0}, n), ldosReference({0}, n), 0.02);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
