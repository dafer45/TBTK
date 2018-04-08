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
	EXPECT_DOUBLE_EQ(solver.getScaleFactor(), 1);
	solver.setScaleFactor(10);
	EXPECT_DOUBLE_EQ(solver.getScaleFactor(), 10);
	solver.setScaleFactor(20);
	EXPECT_DOUBLE_EQ(solver.getScaleFactor(), 20);
}

TEST(ChebyshevExpander, setCalculateCoefficientsOnGPU){
	//Tested through ChebyshevExpander::getCalculateCoefficientsOnGPU().
}

TEST(ChebyshevExpander, getCalculateCoefficientsOnGPU){
	ChebyshevExpander solver;
	EXPECT_FALSE(solver.getCalculateCoefficientsOnGPU());
	solver.setCalculateCoefficientsOnGPU(true);
	EXPECT_TRUE(solver.getCalculateCoefficientsOnGPU());
}

TEST(ChebyshevExpander, setGenerateGreensFunctionsOnGPU){
	//Tested through ChebyshevExpander::getGenerateGreensFunctionsOnGPU().
}

TEST(ChebyshevExpander, getGenerateGreensFunctionsOnGPU){
	ChebyshevExpander solver;
	EXPECT_FALSE(solver.getGenerateGreensFunctionsOnGPU());
	solver.setGenerateGreensFunctionsOnGPU(true);
	EXPECT_TRUE(solver.getGenerateGreensFunctionsOnGPU());
}

TEST(ChebyshevExpander, setNumCoefficients){
	//Tested through ChebyshevExpander::getNumCoefficients().
}

TEST(ChebyshevExpander, getNumCoefficients){
	ChebyshevExpander solver;
	EXPECT_EQ(solver.getNumCoefficients(), 1000);
	solver.setNumCoefficients(2000);
	EXPECT_EQ(solver.getNumCoefficients(), 2000);
}

TEST(ChebyshevExpander, setEnergyResolution){
	//Tested through ChebyshevExpander::getEnergyResolution().
}

TEST(ChebyshevExpander, getEnergyResolution){
	ChebyshevExpander solver;
	EXPECT_EQ(solver.getEnergyResolution(), 1000);
	solver.setEnergyResolution(2000);
	EXPECT_EQ(solver.getEnergyResolution(), 2000);
}

TEST(ChebyshevExpander, setLowerBound){
	//Tested through ChebyshevExpander::getLowerBound().
}

TEST(ChebyshevExpander, getLowerBound){
	ChebyshevExpander solver;
	EXPECT_EQ(solver.getLowerBound(), -1);
	solver.setLowerBound(-2);
	EXPECT_EQ(solver.getLowerBound(), -2);
}

TEST(ChebyshevExpander, setUpperBound){
	//Tested through ChebyshevExpander::getUpperBound().
}

TEST(ChebyshevExpander, getUpperBound){
	ChebyshevExpander solver;
	EXPECT_EQ(solver.getUpperBound(), 1);
	solver.setUpperBound(2);
	EXPECT_EQ(solver.getUpperBound(), 2);
}

TEST(ChebyshevExpander, setUseLookupTable){
	//Tested through ChebyshevExpander::getUseLookupTable().
}

TEST(ChebyshevExpander, getUseLookupTable){
	ChebyshevExpander solver;
	EXPECT_FALSE(solver.getUseLookupTable());
	solver.setUseLookupTable(true);
	EXPECT_TRUE(solver.getUseLookupTable());
}

};	//End of namespace Solver
};	//End of namespace TBTK
