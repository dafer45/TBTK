#include "TBTK/Range.h"
#include "TBTK/Solver/ChebyshevExpander.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

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
	ChebyshevExpander solver;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			solver.setScaleFactor(0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			solver.setScaleFactor(-1);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(ChebyshevExpander, getScaleFactor){
	ChebyshevExpander solver;

	//Default value is 1.1.
	EXPECT_DOUBLE_EQ(solver.getScaleFactor(), 1.1);

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

TEST(ChebyshevExpander, setEnergyWindow0){
	ChebyshevExpander solver;
	solver.setScaleFactor(10);
	solver.setEnergyWindow(Range(-9.999, 9.999, 10));
}

TEST(ChebyshevExpander, setEnergyWindow1){
	ChebyshevExpander solver;
	solver.setScaleFactor(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			solver.setEnergyWindow(Range(-10, 9.999, 10));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(ChebyshevExpander, setEnergyWindow2){
	ChebyshevExpander solver;
	solver.setScaleFactor(10);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			solver.setEnergyWindow(Range(-9.999, 10, 10));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(ChebyshevExpander, getEnergyWindow0){
	ChebyshevExpander solver;
	EXPECT_EQ(solver.getEnergyWindow(), Range(-1, 1, 1000));
}

TEST(ChebyshevExpander, getEnergyWindow1){
	ChebyshevExpander solver;
	solver.setEnergyWindow(Range(-0.1, 0.2, 10));
	EXPECT_EQ(solver.getEnergyWindow(), Range(-0.1, 0.2, 10));
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

TEST(ChebyshevExpander, generateGreensFunction0){
	const double SCALE_FACTOR = 10;
	Range energyWindow(-5, 5, 10);
	std::complex<double> i(0, 1);

	ChebyshevExpander solver;
	solver.setScaleFactor(SCALE_FACTOR);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(false);
	solver.setNumCoefficients(3);
	solver.setEnergyWindow(energyWindow);

	std::vector<std::complex<double>> coefficients = {1, 2, 3};
	std::vector<std::complex<double>> greensFunction
		= solver.generateGreensFunction(
			coefficients,
			ChebyshevExpander::Type::Retarded
		);

	for(unsigned int n = 0; n < energyWindow.getResolution(); n++){
		std::complex<double> reference = -coefficients[0]/2.;
		for(unsigned int c = 1; c < coefficients.size(); c++){
			reference -= coefficients[c]*exp(
				-i*((double)c)*acos(energyWindow[n]/SCALE_FACTOR)
			);
		}
		reference *= 2.*i/sqrt(
			pow(SCALE_FACTOR, 2) - pow(energyWindow[n], 2)
		);
		EXPECT_NEAR(real(greensFunction[n]), real(reference), EPSILON_100);
		EXPECT_NEAR(imag(greensFunction[n]), imag(reference), EPSILON_100);
	}
}

TEST(ChebyshevExpander, generateGreensFunction1){
	const double SCALE_FACTOR = 10;
	Range energyWindow(-5, 5, 10);
	std::complex<double> i(0, 1);

	ChebyshevExpander solver;
	solver.setScaleFactor(SCALE_FACTOR);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(false);
	solver.setNumCoefficients(3);
	solver.setEnergyWindow(energyWindow);

	std::vector<std::complex<double>> coefficients = {1, 2, 3};
	std::vector<std::complex<double>> greensFunction
		= solver.generateGreensFunction(
			coefficients,
			ChebyshevExpander::Type::Advanced
		);

	for(unsigned int n = 0; n < energyWindow.getResolution(); n++){
		std::complex<double> reference = coefficients[0]/2.;
		for(unsigned int c = 1; c < coefficients.size(); c++){
			reference += coefficients[c]*exp(
				i*((double)c)*acos(energyWindow[n]/SCALE_FACTOR)
			);
		}
		reference *= 2.*i/sqrt(
			pow(SCALE_FACTOR, 2) - pow(energyWindow[n], 2)
		);
		EXPECT_NEAR(real(greensFunction[n]), real(reference), EPSILON_100);
		EXPECT_NEAR(imag(greensFunction[n]), imag(reference), EPSILON_100);
	}
}

TEST(ChebyshevExpander, generateGreensFunction2){
	const double SCALE_FACTOR = 10;
	Range energyWindow(-5, 5, 10);
	std::complex<double> i(0, 1);

	ChebyshevExpander solver;
	solver.setScaleFactor(SCALE_FACTOR);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(false);
	solver.setNumCoefficients(3);
	solver.setEnergyWindow(energyWindow);

	std::vector<std::complex<double>> coefficients = {1, 2, 3};
	std::vector<std::complex<double>> greensFunction
		= solver.generateGreensFunction(
			coefficients,
			ChebyshevExpander::Type::Principal
		);

	for(unsigned int n = 0; n < energyWindow.getResolution(); n++){
		std::complex<double> reference = 0;
		for(unsigned int c = 1; c < coefficients.size(); c++){
			reference += coefficients[c]*sin(
				((double)c)*acos(energyWindow[n]/SCALE_FACTOR)
			);
		}
		reference *= 2./sqrt(
			pow(SCALE_FACTOR, 2) - pow(energyWindow[n], 2)
		);
		EXPECT_NEAR(real(greensFunction[n]), real(reference), EPSILON_100);
		EXPECT_NEAR(imag(greensFunction[n]), imag(reference), EPSILON_100);
	}
}

TEST(ChebyshevExpander, generateGreensFunction3){
	const double SCALE_FACTOR = 10;
	Range energyWindow(-5, 5, 10);
	std::complex<double> i(0, 1);

	ChebyshevExpander solver;
	solver.setScaleFactor(SCALE_FACTOR);
	solver.setGenerateGreensFunctionsOnGPU(false);
	solver.setUseLookupTable(false);
	solver.setNumCoefficients(3);
	solver.setEnergyWindow(energyWindow);

	std::vector<std::complex<double>> coefficients = {1, 2, 3};
	std::vector<std::complex<double>> greensFunction
		= solver.generateGreensFunction(
			coefficients,
			ChebyshevExpander::Type::NonPrincipal
		);

	for(unsigned int n = 0; n < energyWindow.getResolution(); n++){
		std::complex<double> reference = coefficients[0]/2.;
		for(unsigned int c = 1; c < coefficients.size(); c++){
			reference += coefficients[c]*cos(
				((double)c)*acos(energyWindow[n]/SCALE_FACTOR)
			);
		}
		reference *= 2.*i/sqrt(
			pow(SCALE_FACTOR, 2) - pow(energyWindow[n], 2)
		);
		EXPECT_NEAR(real(greensFunction[n]), real(reference), EPSILON_100);
		EXPECT_NEAR(imag(greensFunction[n]), imag(reference), EPSILON_100);
	}
}

TEST(ChebyshevExpander, generateGreensFunction4){
	const double SCALE_FACTOR = 10;
	Range energyWindow(-5, 5, 10);
	std::complex<double> i(0, 1);

	for(unsigned int m = 0; m < 4; m++){
		ChebyshevExpander solver[2];
		for(unsigned int n = 0; n < 2; n++){
			solver[n].setScaleFactor(SCALE_FACTOR);
			solver[n].setGenerateGreensFunctionsOnGPU(false);
			solver[n].setUseLookupTable(false);
			solver[n].setNumCoefficients(3);
			solver[n].setEnergyWindow(energyWindow);
		}
		solver[0].setUseLookupTable(false);
		solver[1].setUseLookupTable(true);

		std::vector<std::complex<double>> coefficients = {1, 2, 3};
		ChebyshevExpander::Type type;
		switch(m){
		case 0:
			type = ChebyshevExpander::Type::Retarded;
			break;
		case 1:
			type = ChebyshevExpander::Type::Advanced;
			break;
		case 2:
			type = ChebyshevExpander::Type::Principal;
			break;
		case 3:
			type = ChebyshevExpander::Type::NonPrincipal;
			break;
		}
		std::vector<std::complex<double>> greensFunction0
			= solver[0].generateGreensFunction(
				coefficients,
				type
			);
		std::vector<std::complex<double>> greensFunction1
			= solver[1].generateGreensFunction(
				coefficients,
				type
			);

		for(unsigned int n = 0; n < energyWindow.getResolution(); n++){
			EXPECT_NEAR(
				real(greensFunction0[n]),
				real(greensFunction1[n]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(greensFunction0[n]),
				imag(greensFunction1[n]),
				EPSILON_100
			);
		}
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK
