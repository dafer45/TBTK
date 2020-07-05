#include "TBTK/Solver/Solver.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

TEST(Solver, DynamicTypeInformation){
	Solver solver;
	const DynamicTypeInformation &typeInformation
		=solver.getDynamicTypeInformation();
	EXPECT_EQ(typeInformation.getName(), "Solver::Solver");
	EXPECT_EQ(typeInformation.getNumParents(), 0);
}

TEST(Solver, Constructor){
	//Not testable on its own.
}

TEST(Solver, Destructor){
	//Not testable on its own.
}

TEST(Solver, setModel){
	//Tested through Solver::getModel().
}

TEST(Solver, getModel){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(0, {0}, {0});
	model << HoppingAmplitude(1, {1}, {1});
	model << HoppingAmplitude(2, {2}, {2});
	model.construct();

	Solver solver;
	solver.setModel(model);
	EXPECT_EQ(solver.getModel().getBasisSize(), 3);
	EXPECT_EQ(((const Solver)solver).getModel().getBasisSize(), 3);
}

};	//End of namespace Solver
};	//End of namespace TBTK
