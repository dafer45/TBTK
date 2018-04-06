#include "TBTK/Solver/ArnoldiIterator.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

TEST(ArnoldiIterator, Constructor){
	//Not testable on its own.
}

TEST(ArnoldiIterator, Destructor){
	//Not testable on its own.
}

TEST(ArnoldiIterator, setMode){
	//Tested through ArnoldiIterator::getMode().
}

TEST(ArnoldiIterator, getMode){
	ArnoldiIterator solver;
	solver.setMode(ArnoldiIterator::Mode::Normal);
	EXPECT_EQ(solver.getMode(), ArnoldiIterator::Mode::Normal);
	solver.setMode(ArnoldiIterator::Mode::ShiftAndInvert);
	EXPECT_EQ(solver.getMode(), ArnoldiIterator::Mode::ShiftAndInvert);
}

TEST(ArnoldiIterator, setNumEigenValues){
	//Tested through ArnoldiIterator::getNumEigenValues().
}

TEST(ArnoldiIterator, getNumEigenVectors){
	ArnoldiIterator solver;
	solver.setNumEigenValues(10);
	EXPECT_EQ(solver.getNumEigenValues(), 10);
	solver.setNumEigenValues(20);
	EXPECT_EQ(solver.getNumEigenValues(), 20);
}

};	//End of namespace Solver
};	//End of namespace TBTK
