#include "TBTK/FockStateRule/FockStateRuleSet.h"
#include "TBTK/FockStateRule/DifferenceRule.h"
#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace FockStateRule{

class FockStateRuleSetTest : public ::testing::Test{
protected:
	Model model;
	FockSpace<BitRegister> fockSpace0;
	FockSpace<ExtensiveBitRegister> fockSpace1;

	void SetUp() override{
		for(unsigned int n = 0; n < 4; n++)
			model << HoppingAmplitude(1, {n}, {n});
		model.construct();

		fockSpace0 = FockSpace<BitRegister>(
			&model.getHoppingAmplitudeSet(),
			Statistics::FermiDirac,
			1
		);
		fockSpace1 = FockSpace<ExtensiveBitRegister>(
			&model.getHoppingAmplitudeSet(),
			Statistics::FermiDirac,
			1
		);
	}
};

//TBTKFeature FockStateRule.FockStateRuleSet.clone.1 2019-11-05
TEST_F(FockStateRuleSetTest, isSatisfied1){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	fockState.getBitRegister().setBit(0, 1);
	fockState.getBitRegister().setBit(1, 1);
	fockState.getBitRegister().setBit(2, 1);

	EXPECT_TRUE(fockStateRuleSet.isSatisfied(fockSpace0, fockState));
}

//TBTKFeature FockStateRule.FockStateRuleSet.clone.2 2019-11-05
TEST_F(FockStateRuleSetTest, isSatisfied2){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	fockState.getBitRegister().setBit(0, 1);
	fockState.getBitRegister().setBit(2, 1);
	fockState.getBitRegister().setBit(3, 1);

	EXPECT_FALSE(fockStateRuleSet.isSatisfied(fockSpace0, fockState));
}

//TBTKFeature FockStateRule.FockStateRuleSet.clone.3 2019-11-05
TEST_F(FockStateRuleSetTest, isSatisfied3){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	fockState.getBitRegister().setBit(0, 1);

	EXPECT_FALSE(fockStateRuleSet.isSatisfied(fockSpace0, fockState));
}

//TBTKFeature FockStateRule.FockStateRuleSet.clone.4 2019-11-05
TEST_F(FockStateRuleSetTest, isSatisfied4){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockState<ExtensiveBitRegister> fockState(model.getBasisSize());
	fockState.getBitRegister().setBit(0, 1);
	fockState.getBitRegister().setBit(1, 1);
	fockState.getBitRegister().setBit(2, 1);

	EXPECT_TRUE(fockStateRuleSet.isSatisfied(fockSpace1, fockState));
}

//TBTKFeature FockStateRule.FockStateRuleSet.clone.5 2019-11-05
TEST_F(FockStateRuleSetTest, isSatisfied5){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockState<ExtensiveBitRegister> fockState(model.getBasisSize());
	fockState.getBitRegister().setBit(0, 1);
	fockState.getBitRegister().setBit(2, 1);
	fockState.getBitRegister().setBit(3, 1);

	EXPECT_FALSE(fockStateRuleSet.isSatisfied(fockSpace1, fockState));
}

//TBTKFeature FockStateRule.FockStateRuleSet.clone.6 2019-11-05
TEST_F(FockStateRuleSetTest, isSatisfied6){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockState<ExtensiveBitRegister> fockState(model.getBasisSize());
	fockState.getBitRegister().setBit(0, 1);

	EXPECT_FALSE(fockStateRuleSet.isSatisfied(fockSpace1, fockState));
}

//TBTKFeature FockStateRule.FockStateRuleSet.addFockStateRule.1 2019-11-05
//TBTKFeature FockStateRule.FockStateRuleSet.getSize.1 2019-11-05
TEST(FockStateRuleSet, addFockStateRule){
	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(SumRule({{0}, {1}}, 1));
	fockStateRuleSet.addFockStateRule(SumRule({{1}, {2}}, 1));
	EXPECT_EQ(fockStateRuleSet.getSize(), 2);
}

//TBTKFeature FockStateRule.FockStateRuleSet.operatorEqual.1 2019-11-05
TEST_F(FockStateRuleSetTest, operatorEqual1){
	FockStateRuleSet fockStateRuleSet0;
	fockStateRuleSet0.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet0.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockStateRuleSet fockStateRuleSet1;
	fockStateRuleSet1.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet1.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	EXPECT_EQ(fockStateRuleSet0, fockStateRuleSet1);
}

//TBTKFeature FockStateRule.FockStateRuleSet.operatorEqual.1 2019-11-05
TEST_F(FockStateRuleSetTest, operatorEqual2){
	FockStateRuleSet fockStateRuleSet0;
	fockStateRuleSet0.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet0.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockStateRuleSet fockStateRuleSet1;
	fockStateRuleSet1.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet1.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 2)
	);

	EXPECT_FALSE(fockStateRuleSet0 == fockStateRuleSet1);
}

//TBTKFeature FockStateRule.FockStateRuleSet.operatorMultiplication.1 2019-11-05
TEST_F(FockStateRuleSetTest, operatorMultiplication1){
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize()+1);
	LadderOperator<BitRegister> ladderOperator(
		LadderOperator<BitRegister>::Type::Annihilation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);

	FockStateRuleSet fockStateRuleSet0;
	fockStateRuleSet0.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet0.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockStateRuleSet fockStateRuleSet1;
	fockStateRuleSet1.addFockStateRule(
		ladderOperator*SumRule({{0}, {1}, {2}, {3}}, 3)
	);
	fockStateRuleSet1.addFockStateRule(
		ladderOperator*DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	EXPECT_EQ(ladderOperator*fockStateRuleSet0, fockStateRuleSet1);
}

//TBTKFeature FockStateRule.FockStateRuleSet.operatorMultiplication.2 2019-11-05
TEST_F(FockStateRuleSetTest, operatorMultiplication2){
	FockState<ExtensiveBitRegister> templateState(model.getBasisSize());
	ExtensiveBitRegister fermionMask(model.getBasisSize()+1);
	LadderOperator<ExtensiveBitRegister> ladderOperator(
		LadderOperator<ExtensiveBitRegister>::Type::Annihilation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);

	FockStateRuleSet fockStateRuleSet0;
	fockStateRuleSet0.addFockStateRule(SumRule({{0}, {1}, {2}, {3}}, 3));
	fockStateRuleSet0.addFockStateRule(
		DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	FockStateRuleSet fockStateRuleSet1;
	fockStateRuleSet1.addFockStateRule(
		ladderOperator*SumRule({{0}, {1}, {2}, {3}}, 3)
	);
	fockStateRuleSet1.addFockStateRule(
		ladderOperator*DifferenceRule({{0}, {1}}, {{2}, {3}}, 1)
	);

	EXPECT_EQ(ladderOperator*fockStateRuleSet0, fockStateRuleSet1);
}

};
};
