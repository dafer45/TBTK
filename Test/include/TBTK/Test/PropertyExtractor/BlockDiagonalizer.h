#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/UnitHandler.h"
#include <cmath>

#include "gtest/gtest.h"

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();
const double EPSILON_10000 = 10000*std::numeric_limits<double>::epsilon();
const double CHEMICAL_POTENTIAL = 1;

#define SETUP_MODEL() \
	Model model; \
	model.setVerbose(false); \
	const int SIZE = 10; \
	for(int k = 0; k < SIZE; k++){ \
		model << HoppingAmplitude(k*k-10, {k, 0}, {k, 0}); \
		model << HoppingAmplitude(-k*k+10, {k, 1}, {k, 1}); \
		model << HoppingAmplitude(10, {k, 0}, {k, 1}) + HC; \
	} \
	model.construct(); \
	model.setChemicalPotential(CHEMICAL_POTENTIAL); \
 \
	Model modelDiagonalizer[SIZE]; \
	for(int k = 0; k < SIZE; k++){ \
		modelDiagonalizer[k].setVerbose(false); \
		modelDiagonalizer[k] << HoppingAmplitude(k*k-10, {0}, {0}); \
		modelDiagonalizer[k] << HoppingAmplitude(-k*k+10, {1}, {1}); \
		modelDiagonalizer[k] << HoppingAmplitude(10, {0}, {1}) + HC; \
		modelDiagonalizer[k].construct(); \
		modelDiagonalizer[k].setChemicalPotential( \
			CHEMICAL_POTENTIAL \
		); \
	}

#define SETUP_AND_RUN_SOLVER() \
	Solver::BlockDiagonalizer solver; \
	solver.setVerbose(false); \
	solver.setModel(model); \
	solver.run(); \
 \
	Solver::Diagonalizer solverDiagonalizer[SIZE]; \
	for(int k = 0; k < SIZE; k++){ \
		solverDiagonalizer[k].setVerbose(false); \
		solverDiagonalizer[k].setModel(modelDiagonalizer[k]); \
		solverDiagonalizer[k].run(); \
	}

//TODO
//...
TEST(BlockDiagonalizer, Constructor0){
}

TEST(BlockDiagonalizer, setEnergyWindowReal){
	//Not testable on its own.
}

TEST(BlockDiagonalizer, setEnergyWindowMatsubara){
	//Not testable on its own.
}

TEST(BlockDiagonalizer, getEigenValues){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	BlockDiagonalizer propertyExtractor(solver);
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();

	for(int k = 0; k < SIZE; k++){
		Diagonalizer propertyExtractorDiagonalizer(
			solverDiagonalizer[k]
		);

		Property::EigenValues eigenValuesDiagonalizer
			= propertyExtractorDiagonalizer.getEigenValues();
		EXPECT_NEAR(
			eigenValues(2*k + 0),
			eigenValuesDiagonalizer(0),
			EPSILON_100
		);
		EXPECT_NEAR(
			eigenValues(2*k + 1),
			eigenValuesDiagonalizer(1),
			EPSILON_100
		);
	}
}

TEST(BlockDiagonalizer, getEigenValue0){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	BlockDiagonalizer propertyExtractor(solver);
	for(int k = 0; k < SIZE; k++){
		Diagonalizer propertyExtractorDiagonalizer(
			solverDiagonalizer[k]
		);

		EXPECT_NEAR(
			propertyExtractor.getEigenValue(2*k + 0),
			propertyExtractorDiagonalizer.getEigenValue(0),
			EPSILON_100
		);
		EXPECT_NEAR(
			propertyExtractor.getEigenValue(2*k + 1),
			propertyExtractorDiagonalizer.getEigenValue(1),
			EPSILON_100
		);
	}
}

TEST(BlockDiagonalizer, getEigenValue1){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	BlockDiagonalizer propertyExtractor(solver);
	for(int k = 0; k < SIZE; k++){
		Diagonalizer propertyExtractorDiagonalizer(
			solverDiagonalizer[k]
		);

		EXPECT_NEAR(
			propertyExtractor.getEigenValue({k}, 0),
			propertyExtractorDiagonalizer.getEigenValue(0),
			EPSILON_100
		);
		EXPECT_NEAR(
			propertyExtractor.getEigenValue({k}, 1),
			propertyExtractorDiagonalizer.getEigenValue(1),
			EPSILON_100
		);
	}
}

TEST(BlockDiagonalizer, getAmplitude0){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	BlockDiagonalizer propertyExtractor(solver);
	for(int k = 0; k < SIZE; k++){
		Diagonalizer propertyExtractorDiagonalizer(
			solverDiagonalizer[k]
		);

		for(unsigned int state = 0; state < 2; state++){
			for(int n = 0; n < 2; n++){
				EXPECT_NEAR(
					real(
						propertyExtractor.getAmplitude(
							2*k + state,
							{k, n}
						)
					),
					real(
						propertyExtractorDiagonalizer.getAmplitude(
							state,
							{n}
						)
					),
					EPSILON_100
				);
				EXPECT_NEAR(
					imag(
						propertyExtractor.getAmplitude(
							2*k + state,
							{k, n}
						)
					),
					imag(
						propertyExtractorDiagonalizer.getAmplitude(
							state,
							{n}
						)
					),
					EPSILON_100
				);
			}
		}

		//Check zero entries.
		for(int kp = 0; kp < 2; kp++){
			if(k == kp)
				continue;

			for(unsigned int state = 0; state < 2; state++){
				for(int n = 0; n < 2; n++){
					EXPECT_DOUBLE_EQ(
						real(
							propertyExtractor.getAmplitude(
								2*k + state,
								{kp, n}
							)
						),
						0
					);
					EXPECT_DOUBLE_EQ(
						imag(
							propertyExtractor.getAmplitude(
								2*k + state,
								{kp, n}
							)
						),
						0
					);
				}
			}
		}
	}
}

TEST(BlockDiagonalizer, getAmplitude1){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	BlockDiagonalizer propertyExtractor(solver);
	for(int k = 0; k < SIZE; k++){
		Diagonalizer propertyExtractorDiagonalizer(
			solverDiagonalizer[k]
		);

		for(unsigned int state = 0; state < 2; state++){
			for(int n = 0; n < 2; n++){
				EXPECT_NEAR(
					real(
						propertyExtractor.getAmplitude(
							{k},
							state,
							{n}
						)
					),
					real(
						propertyExtractorDiagonalizer.getAmplitude(
							state,
							{n}
						)
					),
					EPSILON_100
				);
				EXPECT_NEAR(
					imag(
						propertyExtractor.getAmplitude(
							{k},
							state,
							{n}
						)
					),
					imag(
						propertyExtractorDiagonalizer.getAmplitude(
							state,
							{n}
						)
					),
					EPSILON_100
				);
			}
		}
	}
}

TEST(BlockDiagonalizer, calculateWaveFunction){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	//Check when all states are calculated.
	BlockDiagonalizer propertyExtractor(solver);
	Property::WaveFunctions waveFunctions0
		= propertyExtractor.calculateWaveFunctions(
			{{IDX_ALL, IDX_ALL}},
			{IDX_ALL}
		);

	for(int state = 0; state < 2*SIZE; state++){
		for(int k = 0; k < SIZE; k++){
			for(int n = 0; n < 2; n++){
				EXPECT_DOUBLE_EQ(
					real(waveFunctions0({k, n}, state)),
					real(
						propertyExtractor.getAmplitude(
							state,
							{k, n}
						)
					)
				);
				EXPECT_DOUBLE_EQ(
					imag(waveFunctions0({k, n}, state)),
					imag(
						propertyExtractor.getAmplitude(
							state,
							{k, n}
						)
					)
				);
			}
		}
	}

	//Check when some states and some indices are calculated.
	std::vector<int> states = {1, 3, 7};
	std::vector<std::vector<Subindex>> sites = {{0, 0}, {3, 1}, {5, 0}};
	Property::WaveFunctions waveFunctions1
		= propertyExtractor.calculateWaveFunctions(
			{sites[0], sites[1], sites[2]},
			{states[0], states[1], states[2]}
		);
	for(unsigned int n = 0; n < states.size(); n++){
		for(unsigned int c = 0; c < sites.size(); c++){
			EXPECT_DOUBLE_EQ(
				real(waveFunctions1({sites[c]}, states[n])),
				real(waveFunctions0({sites[c]}, states[n]))
			);
			EXPECT_DOUBLE_EQ(
				imag(waveFunctions1({sites[c]}, states[n])),
				imag(waveFunctions0({sites[c]}, states[n]))
			);
		}
	}
	EXPECT_THROW(waveFunctions1({0, 1}, 1), IndexException);
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			waveFunctions1({0, 0}, 0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	::testing::FLAGS_gtest_death_test_style = "fast";
}

TEST(BlockDiagonalizer, calculateGreensFunction){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();

	//Setup patterns to calculate the Green's function for.
	std::vector<Index> patterns;
	for(int k = 0; k < SIZE; k++)
		patterns.push_back({{k, IDX_ALL}, {k, IDX_ALL}});

	////////////////////////////////////////////
	// Advanced and Retarded Green's function //
	////////////////////////////////////////////

	const double LOWER_BOUND = -100;
	const double UPPER_BOUND = 100;
	const int RESOLUTION = 1000;

	for(int n = 0; n < 2; n++){
		//Setup the PropertyExtractor
		BlockDiagonalizer propertyExtractor(solver);
		propertyExtractor.setEnergyWindow(
			LOWER_BOUND,
			UPPER_BOUND,
			RESOLUTION
		);
		propertyExtractor.setEnergyInfinitesimal(200./SIZE);

		//Calculate the Green's function for every block.
		Property::GreensFunction greensFunction;
		if(n == 0){
			greensFunction
				= propertyExtractor.calculateGreensFunction(
					patterns,
					Property::GreensFunction::Type::Advanced
				);
		}
		else{
			greensFunction
				= propertyExtractor.calculateGreensFunction(
					patterns,
					Property::GreensFunction::Type::Retarded
				);
		}

		//Check the energy window values.
		EXPECT_DOUBLE_EQ(greensFunction.getLowerBound(), LOWER_BOUND);
		EXPECT_DOUBLE_EQ(greensFunction.getUpperBound(), UPPER_BOUND);
		EXPECT_EQ(greensFunction.getResolution(), RESOLUTION);

		//Check the calculation of the Green's function against results for the
		//Diagonalizer.
		for(int k = 0; k < SIZE; k++){
			//Setup the PropertyExtractor for the Diagonalizer.
			Diagonalizer propertyExtractorDiagonalizer(
				solverDiagonalizer[k]
			);
			propertyExtractorDiagonalizer.setEnergyWindow(
				LOWER_BOUND,
				UPPER_BOUND,
				RESOLUTION
			);
			propertyExtractorDiagonalizer.setEnergyInfinitesimal(
				200./SIZE
			);

			//Calculate the Green's function for a single block using the
			//Diagonalizer.
			Property::GreensFunction greensFunctionDiagonalizer;
			if(n == 0){
				greensFunctionDiagonalizer
					= propertyExtractorDiagonalizer.calculateGreensFunction(
						{
							{
								Index({IDX_ALL}),
								Index({IDX_ALL})
							}
						},
						Property::GreensFunction::Type::Advanced
					);
			}
			else{
				greensFunctionDiagonalizer
					= propertyExtractorDiagonalizer.calculateGreensFunction(
						{
							{
								Index({IDX_ALL}),
								Index({IDX_ALL})
							}
						},
						Property::GreensFunction::Type::Retarded
					);
			}

			//Perform the check for each value.
			for(unsigned int n = 0; n < RESOLUTION; n++){
				for(int c = 0; c < 2; c++){
					for(int m = 0; m < 2; m++){
						//Real part.
						EXPECT_NEAR(
							real(greensFunction({{k, c}, {k, m}}, n)),
							real(
								greensFunctionDiagonalizer(
									{Index({c}), Index({m})},
									n
								)
							),
							EPSILON_100
						);

						//Imaginary part.
						EXPECT_NEAR(
							imag(greensFunction({{k, c}, {k, m}}, n)),
							imag(
								greensFunctionDiagonalizer(
									{Index({c}), Index({m})},
									n
								)
							),
							EPSILON_100
						);
					}
				}
			}
		}
	}

	////////////////////////////////
	// Matsubara Green's function //
	////////////////////////////////

	//Check the calculation of the Matsubara Green's function against
	//results for the Diagonalizer.
	const double TEMPERATURE = 1;
	const double KT
		= UnitHandler::getConstantInNaturalUnits("k_B")*TEMPERATURE;
	const std::complex<double> FUNDAMENTAL_MATSUBARA_ENERGY
		= std::complex<double>(0, 1)*M_PI*KT;
	model.setTemperature(TEMPERATURE);
	const int LOWER_MATSUBARA_ENERGY_INDEX = -11;
	const int UPPER_MATSUBARA_ENERGY_INDEX = 11;
	const int NUM_MATSUBARA_ENERGIES = 12;

	//Setup the PropertyExtractor.
	BlockDiagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_MATSUBARA_ENERGY_INDEX,
		UPPER_MATSUBARA_ENERGY_INDEX,
		0,
		0
	);

	//Calculate the Green's function for every block.
	Property::GreensFunction greensFunction
		= propertyExtractor.calculateGreensFunction(
			patterns,
			Property::GreensFunction::Type::Matsubara
		);

	//Check the energy window values and the fundamental Matsubara energy.
	EXPECT_EQ(
		greensFunction.getLowerMatsubaraEnergyIndex(),
		LOWER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		greensFunction.getUpperMatsubaraEnergyIndex(),
		UPPER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		greensFunction.getNumMatsubaraEnergies(),
		NUM_MATSUBARA_ENERGIES
	);
	EXPECT_DOUBLE_EQ(
		greensFunction.getFundamentalMatsubaraEnergy(),
		imag(FUNDAMENTAL_MATSUBARA_ENERGY)
	);

	//Check the calculation of the Green's function against results for the
	//Diagonalizer.
	for(int k = 0; k < SIZE; k++){
		//Change the temperature for the Diagonalizer model.
		modelDiagonalizer[k].setTemperature(TEMPERATURE);

		//Setup the PropertyExtractor for the Diagonalizer.
		Diagonalizer propertyExtractorDiagonalizer(
			solverDiagonalizer[k]
		);
		propertyExtractorDiagonalizer.setEnergyWindow(
			LOWER_MATSUBARA_ENERGY_INDEX,
			UPPER_MATSUBARA_ENERGY_INDEX,
			0,
			0
		);

		//Calculate the Green's function for a single block using the
		//Diagonalizer.
		Property::GreensFunction greensFunctionDiagonalizer
			= propertyExtractorDiagonalizer.calculateGreensFunction(
				{{Index({IDX_ALL}), Index({IDX_ALL})}},
				Property::GreensFunction::Type::Matsubara
			);

		//Perform the check for each value.
		for(unsigned int n = 0; n < NUM_MATSUBARA_ENERGIES; n++){
			for(int c = 0; c < 2; c++){
				for(int m = 0; m < 2; m++){
					//Real part.
					EXPECT_NEAR(
						real(greensFunction({{k, c}, {k, m}}, n)),
						real(
							greensFunctionDiagonalizer(
								{Index({c}), Index({m})},
								n
							)
						),
						EPSILON_100
					);

					//Imaginary part.
					EXPECT_NEAR(
						imag(greensFunction({{k, c}, {k, m}}, n)),
						imag(
							greensFunctionDiagonalizer(
								{Index({c}), Index({m})},
								n
							)
						),
						EPSILON_100
					);
				}
			}
		}
	}
}

TEST(BlockDiagonalizer, calculateDOS){
	SETUP_MODEL();
	SETUP_AND_RUN_SOLVER();
	const double LOWER_BOUND = -100;
	const double UPPER_BOUND = 100;
	const int RESOLUTION = 1000;

	//Calculate the DOS.
	BlockDiagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Property::DOS dos = propertyExtractor.calculateDOS();

	//Calculate DOS to compare to using the Diagonalizer.
	Diagonalizer propertyExtractorDiagonalizer(solverDiagonalizer[0]);
	propertyExtractorDiagonalizer.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Property::DOS dosDiagonalizer
		= propertyExtractorDiagonalizer.calculateDOS();
	for(int k = 1; k < SIZE; k++){
		Diagonalizer propertyExtractorDiagonalizer(solverDiagonalizer[k]);
		propertyExtractorDiagonalizer.setEnergyWindow(
			LOWER_BOUND,
			UPPER_BOUND,
			RESOLUTION
		);
		Property::DOS d = propertyExtractorDiagonalizer.calculateDOS();
		for(int n = 0; n < RESOLUTION; n++)
			dosDiagonalizer(n) += d(n);
	}

	//Compare the DOS.
	for(int n = 0; n < RESOLUTION; n++)
		EXPECT_NEAR(dos(n), dosDiagonalizer(n), EPSILON_100);
}

//TODO
//...
TEST(BlockDiagonalizer, calculateExpectationValue){
}

//TODO
//...
TEST(BlockDiagonalizer, calculateDensity){
}

//TODO
//...
TEST(BlockDiagonalizer, calculateMagnetization){
}

//TODO
//...
TEST(BlockDiagonalizer, calculateLDOS){
}

//TODO
//...
TEST(BlockDiagonalizer, calculateSpinPolarizedLDOS){
}

//TODO
//...
TEST(BlockDiagonalizer, calculateEntropy){
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
