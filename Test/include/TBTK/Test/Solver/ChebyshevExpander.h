#include "TBTK/Solver/ChebyshevExpander.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

TEST(ChebyshevExpander, DynamicTypeInformation){
	ChebyshevExpander solver;
	const DynamicTypeInformation &typeInformation
		= solver.getDynamicTypeInformation();
	EXPECT_EQ(typeInformation.getName(), "Solver::ChebyshevExpander");
	EXPECT_EQ(typeInformation.getNumParents(), 1);
	EXPECT_EQ(typeInformation.getParent(0).getName(), "Solver::Solver");
}

TEST(ChebyshevExpander, Constructor){
	//Not testable on its own.
}

TEST(ChebyshevExpander, Destructor){
	//Not testable on its own.
}

//TODO
//This function should not be needed once the HoppingAmplitudeSet is
//restructured to return a SparseMatrix on demand, rather than store a custom
//sparse representation internally.
TEST(ChebyshevExpander, setModel){
}

TEST(ChebyshevExpander, setScaleFactor){
	//Tested through
	//ChebyshevExpander::getScaleFactor().
}

TEST(ChebyshevExpander, getScaleFactor){
	ChebyshevExpander solver;

	//Default value is 1.
	EXPECT_DOUBLE_EQ(solver.getScaleFactor(), 1);

	//Test setting and getting.
	solver.setScaleFactor(10);
	EXPECT_DOUBLE_EQ(solver.getScaleFactor(), 10);
}

TEST(ChebyshevExpander, setNumCoefficients){
	//Tested through ChebyshevExpander::getNumCoefficients().
}

TEST(ChebyshevExpander, getNumCoefficients){
	ChebyshevExpander solver;

	//Default value is 1000.
	EXPECT_EQ(solver.getNumCoefficients(), 1000);

	//Test setting and getting.
	solver.setNumCoefficients(2000);
	EXPECT_EQ(solver.getNumCoefficients(), 2000);
}

TEST(ChebyshevExpander, setBroadening){
	//Tested through ChebyshevExpander::getBroadening().
}

TEST(ChebyshevExpander, getBroadening){
	ChebyshevExpander solver;

	//Default value is 1e-6
	EXPECT_DOUBLE_EQ(solver.getBroadening(), 1e-6);

	//Test setting and getting.
	solver.setBroadening(0.01);
	EXPECT_DOUBLE_EQ(solver.getBroadening(), 0.01);
}

TEST(ChebyshevExpander, setEnergyResolution){
	//Tested through ChebyshevExpander::getEnergyResolution().
}

TEST(ChebyshevExpander, getEnergyResolution){
	ChebyshevExpander solver;

	//Default value is 1000.
	EXPECT_EQ(solver.getEnergyResolution(), 1000);

	//Test setting and getting.
	solver.setEnergyResolution(2000);
	EXPECT_EQ(solver.getEnergyResolution(), 2000);
}

TEST(ChebyshevExpander, setLowerBound){
	//Tested through ChebyshevExpander::getLowerBound().
}

TEST(ChebyshevExpander, getLowerBound){
	ChebyshevExpander solver;

	//Default value is -1.
	EXPECT_EQ(solver.getLowerBound(), -1);

	//Test setting and getting.
	solver.setLowerBound(-2);
	EXPECT_EQ(solver.getLowerBound(), -2);
}

TEST(ChebyshevExpander, setUpperBound){
	//Tested through ChebyshevExpander::getUpperBound().
}

TEST(ChebyshevExpander, getUpperBound){
	ChebyshevExpander solver;

	//Default value is 1.
	EXPECT_EQ(solver.getUpperBound(), 1);

	//Test setting and getting.
	solver.setUpperBound(2);
	EXPECT_EQ(solver.getUpperBound(), 2);
}

TEST(ChebyshevExpander, setCalculateCoefficientsOnGPU){
	//Tested through ChebyshevExpander::getCalculateCoefficientsOnGPU().
}

TEST(ChebyshevExpander, getCalculateCoefficientsOnGPU){
	ChebyshevExpander solver;

	//Default value is false.
	EXPECT_FALSE(solver.getCalculateCoefficientsOnGPU());

	//Test setting and getting.
	solver.setCalculateCoefficientsOnGPU(true);
	EXPECT_TRUE(solver.getCalculateCoefficientsOnGPU());
}

TEST(ChebyshevExpander, setGenerateGreensFunctionsOnGPU){
	//Tested through ChebyshevExpander::getGenerateGreensFunctionsOnGPU().
}

TEST(ChebyshevExpander, getGenerateGreensFunctionsOnGPU){
	ChebyshevExpander solver;

	//Default value is false.
	EXPECT_FALSE(solver.getGenerateGreensFunctionsOnGPU());

	//Test setting and getting.
	solver.setGenerateGreensFunctionsOnGPU(true);
	EXPECT_TRUE(solver.getGenerateGreensFunctionsOnGPU());
}

TEST(ChebyshevExpander, setUseLookupTable){
	//Tested through ChebyshevExpander::getUseLookupTable().
}

TEST(ChebyshevExpander, getUseLookupTable){
	ChebyshevExpander solver;

	//Default value is false.
	EXPECT_FALSE(solver.getUseLookupTable());

	//Test setting and getting.
	solver.setUseLookupTable(true);
	EXPECT_TRUE(solver.getUseLookupTable());
}

TEST(ChebyshevExpander, calculateCoefficients){
	const int SIZE = 5;
	const double mu = -2;
	const double t = 1;
	Model model;
	model.setVerbose(false);
	for(int x = 0; x < SIZE; x++){
		for(int y = 0; y < SIZE; y++){
			model << HoppingAmplitude(-mu, {x,		y},		{x, y});
			model << HoppingAmplitude(-t, {(x+1)%SIZE,	y},		{x, y}) + HC;
			model << HoppingAmplitude(-t, {x,		(y+1)%SIZE},	{x, y}) + HC;
		}
	}
	model.construct();

	const double SCALE_FACTOR = 10;

	ChebyshevExpander solver;
	solver.setVerbose(false);
	solver.setModel(model);
	solver.setScaleFactor(SCALE_FACTOR);
	solver.setCalculateCoefficientsOnGPU(false);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(true);
	solver.setNumCoefficients(100);
	solver.setBroadening(0);

	const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

	//Test diagonal entry.
	std::vector<std::complex<double>> coefficientsCPU0
		= solver.calculateCoefficients({0, 0}, {0, 0});
	EXPECT_EQ(coefficientsCPU0.size(), 100);
	//<j_t|j_f>
	EXPECT_NEAR(real(coefficientsCPU0[0]), 1, EPSILON_100);
	EXPECT_NEAR(imag(coefficientsCPU0[0]), 0, EPSILON_100);
	//<j_t|H|j_f>
	EXPECT_NEAR(real(coefficientsCPU0[1]), -mu/SCALE_FACTOR, EPSILON_100);
	EXPECT_NEAR(imag(coefficientsCPU0[1]), 0, EPSILON_100);
	//<j_t|(2H^2 - I)|j_f>
	EXPECT_NEAR(
		real(coefficientsCPU0[2]),
		2*(mu*mu + 4*t*t)/(SCALE_FACTOR*SCALE_FACTOR) - 1,
		EPSILON_100
	);
	EXPECT_NEAR(imag(coefficientsCPU0[2]), 0, EPSILON_100);

	//Test off-diagonal entry.
	std::vector<std::complex<double>> coefficientsCPU1
		= solver.calculateCoefficients({1, 0}, {0, 0});
	EXPECT_EQ(coefficientsCPU1.size(), 100);
	EXPECT_NEAR(real(coefficientsCPU1[0]), 0, EPSILON_100);
	EXPECT_NEAR(imag(coefficientsCPU1[0]), 0, EPSILON_100);
	//<j_t|H|j_f>
	EXPECT_NEAR(real(coefficientsCPU1[1]), -t/SCALE_FACTOR, EPSILON_100);
	EXPECT_NEAR(imag(coefficientsCPU1[1]), 0, EPSILON_100);
	//<j_t|(2H^2 - I)|j_f>
	EXPECT_NEAR(
		real(coefficientsCPU1[2]),
		4*mu*t/(SCALE_FACTOR*SCALE_FACTOR),
		EPSILON_100
	);
	EXPECT_NEAR(imag(coefficientsCPU1[2]), 0, EPSILON_100);

	//Test multiple to-Indices at once.
	std::vector<Index> toIndices;
	toIndices.push_back({0, 0});
	toIndices.push_back({1, 0});
	std::vector<std::vector<std::complex<double>>> coefficientsCPU2
		= solver.calculateCoefficients(toIndices, {0, 0});
	ASSERT_EQ(coefficientsCPU2[0].size(), 100);
	ASSERT_EQ(coefficientsCPU2[1].size(), 100);
	for(unsigned int n = 0; n < 100; n++){
		EXPECT_NEAR(
			real(coefficientsCPU2[0][n]),
			real(coefficientsCPU0[n]),
			EPSILON_100
		);
		EXPECT_NEAR(
			imag(coefficientsCPU2[0][n]),
			imag(coefficientsCPU0[n]),
			EPSILON_100
		);
		EXPECT_NEAR(
			real(coefficientsCPU2[1][n]),
			real(coefficientsCPU1[n]),
			EPSILON_100
		);
		EXPECT_NEAR(
			imag(coefficientsCPU2[1][n]),
			imag(coefficientsCPU1[n]),
			EPSILON_100
		);
	}

	solver.setCalculateCoefficientsOnGPU(true);
	#ifdef TBTK_CUDA_ENABLED
//		model.constructCOO();
		Communicator::setGlobalVerbose(false);

		//Test diagonal entry.
		std::vector<std::complex<double>> coefficientsGPU0
			= solver.calculateCoefficients({0, 0}, {0, 0});
		EXPECT_EQ(coefficientsGPU0.size(), 100);

		//Test off-diagonal entry.
		std::vector<std::complex<double>> coefficientsGPU1
			= solver.calculateCoefficients({1, 0}, {0, 0});
		EXPECT_EQ(coefficientsGPU1.size(), 100);

		//Test multiple to-Indices at once.
		std::vector<std::vector<std::complex<double>>> coefficientsGPU2
			= solver.calculateCoefficients(toIndices, {0, 0});
		ASSERT_EQ(coefficientsGPU2[0].size(), 100);
		ASSERT_EQ(coefficientsGPU2[1].size(), 100);
		for(unsigned int n = 0; n < 100; n++){
			EXPECT_NEAR(
				real(coefficientsGPU0[n]),
				real(coefficientsCPU0[n]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(coefficientsGPU0[n]),
				imag(coefficientsCPU0[n]),
				EPSILON_100
			);

			EXPECT_NEAR(
				real(coefficientsGPU1[n]),
				real(coefficientsCPU1[n]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(coefficientsGPU1[n]),
				imag(coefficientsCPU1[n]),
				EPSILON_100
			);

			EXPECT_NEAR(
				real(coefficientsGPU2[0][n]),
				real(coefficientsCPU0[n]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(coefficientsGPU2[0][n]),
				imag(coefficientsCPU0[n]),
				EPSILON_100
			);

			EXPECT_NEAR(
				real(coefficientsGPU2[1][n]),
				real(coefficientsCPU1[n]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(coefficientsGPU2[1][n]),
				imag(coefficientsCPU1[n]),
				EPSILON_100
			);
		}
	#else
		EXPECT_EXIT(
			{
				Streams::setStdMuteErr();
				std::vector<std::complex<double>> coefficientsGPU0
					= solver.calculateCoefficients({0, 0}, {0, 0});
			},
			::testing::ExitedWithCode(1),
			""
		);
		EXPECT_EXIT(
			{
				Streams::setStdMuteErr();
				std::vector<
					std::vector<std::complex<double>>
				> coefficientsGPU2
					= solver.calculateCoefficients(
						toIndices,
						{0, 0}
					);
			},
			::testing::ExitedWithCode(1),
			""
		);
	#endif
}

//TODO
//...
TEST(ChebyshevExpander, generateGreensFunction){
}

};	//End of namespace Solver
};	//End of namespace TBTK
