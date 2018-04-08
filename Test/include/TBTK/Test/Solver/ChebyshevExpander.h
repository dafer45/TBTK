#include "TBTK/Solver/ChebyshevExpander.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

//TODO
//Restructure the ChebyshevExpander to hide CPU/GPU specific functions from the
//public interface and to move method specific detils from the
//PropertyExtractor::ChebyshevExpander to Solver::ChebyshevExpander. Implement
//tests on the resulting public interface.

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

};	//End of namespace Solver
};	//End of namespace TBTK
