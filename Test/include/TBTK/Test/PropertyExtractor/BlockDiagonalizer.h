#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
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

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
