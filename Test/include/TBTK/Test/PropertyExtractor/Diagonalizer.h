#include "TBTK/IndexException.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/UnitHandler.h"
#include <cmath>
#include <complex>

#include "gtest/gtest.h"

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();
const double EPSILON_10000 = 10000*std::numeric_limits<double>::epsilon();
const double CHEMICAL_POTENTIAL = 1;

#define SETUP_MODEL() \
	Model model; \
	model.setVerbose(false); \
	const int SIZE = 50; \
	for(int x = 0; x < SIZE; x++) \
		model << HoppingAmplitude(-1, {(x+1)%SIZE}, {x}) + HC; \
	model.construct(); \
	model.setChemicalPotential(CHEMICAL_POTENTIAL);

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

TEST(Diagonalizer, calculateGreensFunction){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(-5, 5, 100);
	double delta = 0.1;
	propertyExtractor.setEnergyInfinitesimal(delta);

	//Verify that both the Advanced and Retarded Green's function is
	//calculated correctly.
	for(unsigned int n = 0; n < 2; n++){
		Property::GreensFunction greensFunction;
		double sign;
		switch(n){
		case 0:
			greensFunction
				= propertyExtractor.calculateGreensFunction(
					{{Index({IDX_ALL}), Index({0})}},
					Property::GreensFunction::Type::Advanced
				);
			sign = -1;

			break;
		case 1:
			greensFunction
				= propertyExtractor.calculateGreensFunction(
					{{Index({IDX_ALL}), Index({0})}},
					Property::GreensFunction::Type::Retarded
				);
			sign = 1;

			break;
		default:
			TBTKExit(
				"Test::PropertyExtractor::Diagonalizer::calculateGreensFunction()",
				"Unknown action for n='" << n << "'.",
				"This should never happen, contact the"
				<< " developer."
			);
		}

		EXPECT_DOUBLE_EQ(greensFunction.getLowerBound(), -5);
		EXPECT_DOUBLE_EQ(greensFunction.getUpperBound(), 5);
		EXPECT_EQ(greensFunction.getResolution(), 100);

		std::complex<double> i(0, 1);
		for(int x = 0; x < SIZE; x++){
			for(int n = 0; n < 100; n++){
				std::complex<double> gf = 0;
				double E = -5 + 0.1*n;
				for(unsigned int c = 0; c < SIZE; c++){
					double E_c
						= propertyExtractor.getEigenValue(c);
					std::complex<double> amplitude0
						= propertyExtractor.getAmplitude(
							c, {x}
						);
					std::complex<double> amplitude1
						= propertyExtractor.getAmplitude(
							c, {0}
						);
					gf += amplitude0*conj(amplitude1)/(
						E - E_c + sign*i*delta
					);
				}

				EXPECT_NEAR(
					real(
						greensFunction(
							{
								Index({x}),
								Index({0})
							},
							n
						)
					),
					real(gf),
					EPSILON_100
				);
				EXPECT_NEAR(
					imag(
						greensFunction(
							{
								Index({x}),
								Index({0})
							},
							n
						)
					),
					imag(gf),
					EPSILON_100
				);
			}
		}
	}

	//Verify that the Matsubara Green's function is calculated correctly.
	const double TEMPERATURE = 1;
	const double KT = UnitHandler::getK_BN()*TEMPERATURE;
	const double FUNDAMENTAL_MATSUBARA_ENERGY = M_PI*KT;
	model.setTemperature(TEMPERATURE);
	const int LOWER_MATSUBARA_ENERGY_INDEX = -11;
	const int UPPER_MATSUBARA_ENERGY_INDEX = 11;
	const int NUM_MATSUBARA_ENERGIES = 12;
	propertyExtractor.setEnergyWindow(
		LOWER_MATSUBARA_ENERGY_INDEX,
		UPPER_MATSUBARA_ENERGY_INDEX,
		0,
		0
	);
	Property::GreensFunction greensFunction
		= propertyExtractor.calculateGreensFunction(
			{{Index({IDX_ALL}), Index({0})}},
			Property::GreensFunction::Type::Matsubara
		);

	EXPECT_EQ(
		greensFunction.getLowerMatsubaraEnergyIndex(),
		LOWER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		greensFunction.getUpperMatsubaraEnergyIndex(),
		UPPER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_DOUBLE_EQ(
		greensFunction.getFundamentalMatsubaraEnergy(),
		FUNDAMENTAL_MATSUBARA_ENERGY
	);

	std::complex<double> i(0, 1);
	for(int x = 0; x < SIZE; x++){
		for(int n = 0; n < NUM_MATSUBARA_ENERGIES; n++){
			std::complex<double> gf = 0;
/*			double temperature = model.getTemperature();
			double kT = UnitHandler::getK_BN()*temperature;
			std::complex<double> E = (
				LOWER_MATSUBARA_ENERGY_INDEX + 2.*n
			)*i*M_PI*kT;*/
			std::complex<double> E = (
				LOWER_MATSUBARA_ENERGY_INDEX + 2.*n
			)*i*FUNDAMENTAL_MATSUBARA_ENERGY;
			for(unsigned int c = 0; c < SIZE; c++){
				double E_c
					= propertyExtractor.getEigenValue(c);
				std::complex<double> amplitude0
					= propertyExtractor.getAmplitude(
						c, {x}
					);
				std::complex<double> amplitude1
					= propertyExtractor.getAmplitude(
						c, {0}
					);
				gf += amplitude0*conj(amplitude1)/(
					E - E_c
				);
			}

			EXPECT_NEAR(
				real(
					greensFunction(
						{
							Index({x}),
							Index({0})
						},
						n
					)
				),
				real(gf),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(
					greensFunction(
						{
							Index({x}),
							Index({0})
						},
						n
					)
				),
				imag(gf),
				EPSILON_100
			);
		}
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

TEST(Diagonalizer, calculateDOS){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();
	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const int RESOLUTION = 1000;

	Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	//Calculate DOS to compare to. EigenValues has already been checked
	//using Diagonalizer::getEigenValues(), so it is safe to use it here
	//for these this test.
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();
	double dosBenchmark[1000];
	for(unsigned int n = 0; n < RESOLUTION; n++)
		dosBenchmark[n] = 0.;
	double dE = (UPPER_BOUND - LOWER_BOUND)/RESOLUTION;
	for(unsigned int n = 0; n < SIZE; n++){
		int e = (int)(
			RESOLUTION*(eigenValues(n) - LOWER_BOUND)/(
				UPPER_BOUND - LOWER_BOUND
			)
		);
		if(e >= 0 && e < RESOLUTION)
			dosBenchmark[e] += 1/dE;
	}

	//Calculate DOS
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Check that bounds and resolution are corectly set.
	EXPECT_DOUBLE_EQ(dos.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(dos.getUpperBound(), 10);
	ASSERT_EQ(dos.getResolution(), 1000);

	//Check that the dos agrees with the benchmark and that it integrates
	//to the number of states in the Model.
	double integratedDOS = 0;
	for(unsigned int n = 0; n < RESOLUTION; n++){
		integratedDOS += dos(n)*dE;
		EXPECT_DOUBLE_EQ(dos(n), dosBenchmark[n]);
	}
	EXPECT_NEAR(integratedDOS, SIZE, EPSILON_100);
}

//TODO
//...
TEST(Diagonalizer, calculateExpectationValue){
}

TEST(Diagonalizer, calculateDensity){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	Diagonalizer propertyExtractor(solver);

	//Calculate density to compare to. getEigenValue() and getAmplitude()
	//has already been checked using Diagonalizer::getEigenValue() and
	//Diagonalizer::getAmplitude(), so it is safe to use them here for this
	//test.
	double densityBenchmark = 0;
	for(unsigned int n = 0; n < SIZE; n++){
		if(propertyExtractor.getEigenValue(n) - CHEMICAL_POTENTIAL > 0)
			continue;
		densityBenchmark += pow(
			abs(propertyExtractor.getAmplitude(n, {0})
			), 2
		);
	}

	//Check Ranges format.
	Property::Density density0 = propertyExtractor.calculateDensity(
		{IDX_X},
		{SIZE}
	);
	ASSERT_EQ(density0.getSize(), SIZE);
	const std::vector<double> &data = density0.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_NEAR(data[n], densityBenchmark, EPSILON_100);

	//Check Custom format.
	Property::Density density1 = propertyExtractor.calculateDensity({
		{IDX_ALL}
	});
	ASSERT_EQ(density1.getSize(), SIZE);
	for(unsigned int n = 0; n < density1.getSize(); n++)
		EXPECT_NEAR(density1({n}), densityBenchmark, EPSILON_100);
}

//TODO
//...
TEST(Diagonalizer, calculateMagnetization){
}

TEST(Diagonalizer, calculateLDOS){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();
	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const int RESOLUTION = 1000;

	Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	Property::DOS dos = propertyExtractor.calculateDOS();

	//Check Ranges format.
	Property::LDOS ldos0 = propertyExtractor.calculateLDOS(
		{IDX_X},
		{SIZE}
	);
	const std::vector<double> &data = ldos0.getData();
	for(int n = 0; n < ldos0.getResolution(); n++){
		for(int x = 0; x < SIZE; x++){
			EXPECT_NEAR(
				data[RESOLUTION*x + n],
				dos(n)/SIZE,
				EPSILON_10000
			);
		}
	}

	//Check Custom format.
	Property::LDOS ldos1 = propertyExtractor.calculateLDOS({{IDX_ALL}});
	for(unsigned int n = 0; n < RESOLUTION; n++)
		for(int x = 0; x < SIZE; x++)
			EXPECT_NEAR(ldos1({x}, n), dos(n)/SIZE, EPSILON_10000);
}

//TODO
//...
TEST(Diagonalizer, calculateSpinPolarizedLDOS){
}

//TODO
//...
TEST(Diagonalizer, calculateEntropy){
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
