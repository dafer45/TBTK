#include "TBTK/IndexException.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include <cmath>

#include "gtest/gtest.h"

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

#define SETUP_MODEL() \
	Model model; \
	model.setVerbose(false); \
	const int SIZE = 50; \
	for(int x = 0; x < SIZE; x++) \
		model << HoppingAmplitude(-1, {(x+1)%SIZE}, {x}) + HC; \
	model.construct();

#define SETUP_AND_RUN_SOLVER() \
	Solver::Diagonalizer solver; \
	solver.setVerbose(false); \
	solver.setModel(model); \
	solver.run();

#define SETUP_ANALYTICAL_EIGEN_VALUES() \
	std::vector<double> analyticalEigenValues; \
	for(unsigned int n = 0; n < SIZE; n++) \
		analyticalEigenValues.push_back(-2*cos(2*M_PI*n/SIZE)); \
	std::sort( \
		analyticalEigenValues.begin(), \
		analyticalEigenValues.end() \
	);

//TODO
//...
TEST(Diagonalizer, Constructor0){
}

TEST(Diagonalizer, getEigenValues){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();
	SETUP_ANALYTICAL_EIGEN_VALUES();

	Diagonalizer propertyExtractor(solver);
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();

	for(unsigned int n = 0; n < SIZE; n++)
		EXPECT_NEAR(eigenValues(n), analyticalEigenValues[n], EPSILON_100);
}

TEST(Diagonalizer, getEigenValue){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();
	SETUP_ANALYTICAL_EIGEN_VALUES();

	Diagonalizer propertyExtractor(solver);
	for(unsigned int n = 0; n < SIZE; n++){
		EXPECT_NEAR(
			propertyExtractor.getEigenValue(n),
			analyticalEigenValues[n],
			EPSILON_100
		);
	}
}

TEST(Diagonalizer, getAmplitude){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	Diagonalizer propertyExtractor(solver);

	//Check that the states are normalized.
	for(unsigned int n = 0; n < SIZE; n++){
		double totalProbability = 0;
		for(int x = 0; x < SIZE; x++){
			std::complex<double> amplitude
				= propertyExtractor.getAmplitude(n, {x});
			totalProbability += pow(abs(amplitude), 2);
		}
		EXPECT_NEAR(totalProbability, 1, EPSILON_100);
	}

	//The lowest energy state is in the subspace spanned by 1, the next two
	//states are in the subspace spanned by cos(x) and sin(x), the third
	//and fourth states are in the subspace spanned by cos(2x) and sin(2x),
	//and so forth. Verify this by checking that the corresponding states
	//project fully onto these subspaces.
	double subspaceBases[SIZE][2][SIZE];
	for(unsigned int n = 0; n < SIZE; n++){
		double normalizationFactor = 1/5.;
		//The first and last states are in spaces spanned by a single
		//state while all other are in a subspace spanned by two
		//states. To simplify the check, the subspaces for the first
		//and last subspace is artificially given "two dimensions" by
		//replicating half of the state twice in subspaceBases. The
		//sin() state is automatically set to zero and therefore
		//automatically falls out of tha calculation. However, the
		//reamining state needs a different normalization factor
		if(n == 0 || n == SIZE-1)
			normalizationFactor /= sqrt(2.);

		for(unsigned int x = 0; x < SIZE; x++){
			subspaceBases[n][0][x]
				= normalizationFactor*cos(
					2*M_PI*((n+1)/2)*(x/(double)SIZE)
				);
		}
		for(unsigned int x = 0; x < SIZE; x++){
			subspaceBases[n][1][x]
				= normalizationFactor*sin(
					2*M_PI*((n+1)/2)*(x/(double)SIZE)
				);
		}
	}

	for(unsigned int n = 0; n < SIZE; n++){
		std::complex<double> projectionAmplitude0 = 0;
		std::complex<double> projectionAmplitude1 = 0;
		for(int x = 0; x < SIZE; x++){
			std::complex<double> amplitude
				= propertyExtractor.getAmplitude(n, {x});

			projectionAmplitude0 += subspaceBases[n][0][x]*amplitude;
			projectionAmplitude1 += subspaceBases[n][1][x]*amplitude;
		}
		double projectionAmplitudeTotal = sqrt(
			pow(std::abs(projectionAmplitude0), 2.)
			+ pow(std::abs(projectionAmplitude1), 2.)
		);
		EXPECT_NEAR(std::abs(projectionAmplitudeTotal), 1, EPSILON_100);
	}
}

TEST(Diagonalizer, calculateWaveFunctions){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	Diagonalizer propertyExtractor(solver);

	//Check when all states are calculated.
	std::vector<unsigned int> states0;
	for(unsigned int n = 0; n < SIZE; n++)
		states0.push_back(n);
	Property::WaveFunctions waveFunctions0
		= propertyExtractor.calculateWaveFunctions(
			{{IDX_ALL}},
			{IDX_ALL}
		);

	for(unsigned int n = 0; n < SIZE; n++){
		for(int x = 0; x < SIZE; x++){
			EXPECT_DOUBLE_EQ(
				real(waveFunctions0({x}, n)),
				real(propertyExtractor.getAmplitude(n, {x}))
			);
			EXPECT_DOUBLE_EQ(
				imag(waveFunctions0({x}, n)),
				imag(propertyExtractor.getAmplitude(n, {x}))
			);
		}
	}

	//Check when some states and some indices are calculated.
	std::vector<int> states1 = {1, 3, 7};
	std::vector<int> sites1 = {11, 13, 19};
	Property::WaveFunctions waveFunctions1
		= propertyExtractor.calculateWaveFunctions(
			{{sites1[0]}, {sites1[1]}, {sites1[2]}},
			states1
		);
	for(unsigned int n = 0; n < states1.size(); n++){
		for(unsigned int x = 0; x < sites1.size(); x++){
			EXPECT_DOUBLE_EQ(
				real(waveFunctions1({sites1[x]}, states1[n])),
				real(waveFunctions0({sites1[x]}, states1[n]))
			);
			EXPECT_DOUBLE_EQ(
				imag(waveFunctions1({sites1[x]}, states1[n])),
				imag(waveFunctions0({sites1[x]}, states1[n]))
			);
		}
	}
	EXPECT_THROW(waveFunctions1({12}, 3), IndexException);
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1({11}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	::testing::FLAGS_gtest_death_test_style = "fast";
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
