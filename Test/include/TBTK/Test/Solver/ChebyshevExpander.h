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

};	//End of namespace Solver
};	//End of namespace TBTK
