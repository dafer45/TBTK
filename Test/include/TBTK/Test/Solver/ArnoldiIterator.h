#include "TBTK/Solver/ArnoldiIterator.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

TEST(ArnoldiIterator, DynamicTypeInformation){
	ArnoldiIterator solver;
	const DynamicTypeInformation &typeInformation
		= solver.getDynamicTypeInformation();
	EXPECT_EQ(typeInformation.getName(), "Solver::ArnoldiIterator");
	EXPECT_EQ(typeInformation.getNumParents(), 1);
	EXPECT_EQ(typeInformation.getParent(0).getName(), "Solver::Solver");
}

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

TEST(ArnoldiIterator, setCalculateEigenVectors){
	//Tested through ArnoldiIterator::getCalculateEigenVectors().
}

TEST(ArnoldiIterator, getCalculateEigenVectors){
	ArnoldiIterator solver;
	EXPECT_FALSE(solver.getCalculateEigenVectors());
	solver.setCalculateEigenVectors(true);
	EXPECT_TRUE(solver.getCalculateEigenVectors());
}

TEST(ArnoldiIterator, setNumLanczosVectors){
	//Tested through ArnoldiIterator::getNumLnczosVectors().
}

TEST(ArnoldiIterator, getNumLanczosVectors){
	ArnoldiIterator solver;
	solver.setNumLanczosVectors(10);
	EXPECT_EQ(solver.getNumLanczosVectors(), 10);
	solver.setNumLanczosVectors(20);
	EXPECT_EQ(solver.getNumLanczosVectors(), 20);
}

TEST(ArnoldiIterator, setTolerance){
	//TEST NOT CLEAR: It is not clear how this parameter can be tested
	//since it is a tuning parameter for ARPACK.
}

TEST(ArnoldiIterator, setMaxIterations){
	//TEST NOT CLEAR: It is not clear how this parameter can be tested
	//since it is a stoping criterion for an internal loop hidden away from
	//the public interface.
}

TEST(ArnoldiIterator, setCentralValue){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {1}, {0}) + HC;
	model << HoppingAmplitude(2, {2}, {2});
	model.construct();
//	model.constructCOO();

	ArnoldiIterator solver;
	solver.setVerbose(false);
	solver.setModel(model);
	solver.setNumEigenValues(1);
	solver.setNumLanczosVectors(3);
	solver.setMaxIterations(10);

	solver.setMode(ArnoldiIterator::Mode::Normal);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 2, EPSILON_100);
	solver.setCentralValue(3);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), -1, EPSILON_100);
	solver.setCentralValue(-0.5);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 2, EPSILON_100);

	solver.setMode(ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(-1.1);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), -1, 1e-5);
	solver.setCentralValue(-0.9);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), -1, 1e-5);
	solver.setCentralValue(-0.1);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), -1, 1e-5);
	solver.setCentralValue(0.1);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 1, 1e-1);
	solver.setCentralValue(0.9);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 1, 1e-5);
	solver.setCentralValue(1.1);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 1, 1e-5);
	solver.setCentralValue(1.9);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 2, 1e-5);
	solver.setCentralValue(2.1);
	solver.run();
	EXPECT_NEAR(solver.getEigenValue(0), 2, 1e-5);
}

TEST(ArnoldiIterator, run){
	//Already tested through
	//ArnoldiIterator::setCentralValue()
	//ArnoldiIterator::getEigenValues()
	//ArnoldiIterator::getAmplitude()
}

TEST(ArnoldiIterator, getEigenValues){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {1}, {0}) + HC;
	model << HoppingAmplitude(2, {2}, {2});
	model << HoppingAmplitude(3, {3}, {3});
	model << HoppingAmplitude(3, {4}, {4});
	model << HoppingAmplitude(0.1, {4}, {3}) + HC;
	model << HoppingAmplitude(5, {5}, {5});
	model << HoppingAmplitude(6, {6}, {6});
	model.construct();
//	model.constructCOO();

	ArnoldiIterator solver;
	solver.setVerbose(false);
	solver.setModel(model);
	solver.setNumEigenValues(5);
	solver.setNumLanczosVectors(7);
	solver.setMaxIterations(10);

	solver.setMode(ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(-2);
	solver.run();
	const CArray<std::complex<double>> &eigenValues
		= solver.getEigenValues();
	EXPECT_NEAR(real(eigenValues[0]), -1, 1e-5);
	EXPECT_NEAR(imag(eigenValues[0]), 0, EPSILON_100);
	EXPECT_NEAR(real(eigenValues[1]), 1, 1e-5);
	EXPECT_NEAR(imag(eigenValues[1]), 0, EPSILON_100);
	EXPECT_NEAR(real(eigenValues[2]), 2, 1e-5);
	EXPECT_NEAR(imag(eigenValues[2]), 0, EPSILON_100);
	EXPECT_NEAR(real(eigenValues[3]), 2.9, 1e-5);
	EXPECT_NEAR(imag(eigenValues[3]), 0, EPSILON_100);
	EXPECT_NEAR(real(eigenValues[4]), 3.1, 1e-5);
	EXPECT_NEAR(imag(eigenValues[4]), 0, EPSILON_100);
}

TEST(ArnoldiIterator, getEigenValue){
	//Already tested through
	//ArnoldiIterator::setCentralValue()
}

TEST(ArnoldiIterator, getAmplitude){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {1}, {0}) + HC;
	model << HoppingAmplitude(2, {2}, {2});
	model << HoppingAmplitude(3, {3}, {3});
	model << HoppingAmplitude(3, {4}, {4});
	model << HoppingAmplitude(0.1, {4}, {3}) + HC;
	model << HoppingAmplitude(5, {5}, {5});
	model << HoppingAmplitude(6, {6}, {6});
	model.construct();
//	model.constructCOO();

	ArnoldiIterator solver;
	solver.setVerbose(false);
	solver.setModel(model);
	solver.setNumEigenValues(5);
	solver.setNumLanczosVectors(7);
	solver.setMaxIterations(10);

	solver.setMode(ArnoldiIterator::Mode::ShiftAndInvert);
	solver.setCentralValue(-2);
	solver.setCalculateEigenVectors(true);
	solver.run();

	EXPECT_NEAR(real(solver.getAmplitude(0, {0})/solver.getAmplitude(0, {1})), -1, 1e-5);
	EXPECT_NEAR(imag(solver.getAmplitude(0, {0})/solver.getAmplitude(0, {1})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(0, {2})), 0, EPSILON_100);
	EXPECT_NEAR(imag(solver.getAmplitude(0, {2})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(0, {3})), 0, EPSILON_100);
	EXPECT_NEAR(imag(solver.getAmplitude(0, {3})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(0, {4})), 0, EPSILON_100);
	EXPECT_NEAR(imag(solver.getAmplitude(0, {4})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(0, {5})), 0, EPSILON_100);
	EXPECT_NEAR(imag(solver.getAmplitude(0, {5})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(0, {6})), 0, EPSILON_100);
	EXPECT_NEAR(imag(solver.getAmplitude(0, {6})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(1, {0})/solver.getAmplitude(1, {1})), 1, 1e-5);
	EXPECT_NEAR(imag(solver.getAmplitude(1, {0})/solver.getAmplitude(1, {1})), 0, EPSILON_100);
	EXPECT_NEAR(abs(solver.getAmplitude(2, {2})), 1, 1e-5);
	EXPECT_NEAR(real(solver.getAmplitude(3, {3})/solver.getAmplitude(3, {4})), -1, 1e-5);
	EXPECT_NEAR(imag(solver.getAmplitude(3, {3})/solver.getAmplitude(3, {4})), 0, EPSILON_100);
	EXPECT_NEAR(real(solver.getAmplitude(4, {3})/solver.getAmplitude(4, {4})), 1, 1e-5);
	EXPECT_NEAR(imag(solver.getAmplitude(4, {3})/solver.getAmplitude(4, {4})), 0, EPSILON_100);
}

};	//End of namespace Solver
};	//End of namespace TBTK
