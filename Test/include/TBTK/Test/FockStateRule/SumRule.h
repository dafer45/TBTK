#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace FockStateRule{

//TBTKFeature FockStateRule.SumRule.clone.1 2019-11-05
TEST(SumRule, clone1){
	SumRule sumRule(
		{{0}, {1}},
		1
	);

	SumRule *clone = sumRule.clone();
	EXPECT_EQ(*clone, sumRule);

	delete clone;
}

//TBTKFeature FockStateRule.SumRule.createNewRule.1 2019-11-05
TEST(SumRule, createNewRule1){
	Model model;
	for(unsigned int n = 0; n < 4; n++)
		model << HoppingAmplitude(1, {n}, {n});
	model.construct();
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize()+1);

	LadderOperator<BitRegister> ladderOperator(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);

	SumRule sumRule0(
		{{0}, {1}},
		0
	);
	SumRule sumRule1(
		{{0}, {1}},
		1
	);

	EXPECT_EQ(
		sumRule0.createNewRule(ladderOperator),
		sumRule1
	);
}

//TBTKFeature FockStateRule.SumRule.createNewRule.2 2019-11-05
TEST(SumRule, createNewRule2){
	Model model;
	for(unsigned int n = 0; n < 4; n++)
		model << HoppingAmplitude(1, {n}, {n});
	model.construct();
	FockState<ExtensiveBitRegister> templateState(model.getBasisSize());
	ExtensiveBitRegister fermionMask(model.getBasisSize()+1);

	LadderOperator<ExtensiveBitRegister> ladderOperator(
		LadderOperator<ExtensiveBitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);

	SumRule sumRule0(
		{{0}, {1}},
		0
	);
	SumRule sumRule1(
		{{0}, {1}},
		1
	);

	EXPECT_EQ(sumRule0.createNewRule(ladderOperator), sumRule1);
}

//TBTKFeature FockStateRule.SumRule.isSatisfied.1 2019-11-05
TEST(SumRule, isSatisfied1){
	Model model;
	for(unsigned int n = 0; n < 4; n++)
		model << HoppingAmplitude(1, {n}, {n});
	model.construct();

	FockSpace<BitRegister> fockSpace(
		&model.getHoppingAmplitudeSet(),
		Statistics::FermiDirac,
		1
	);

	FockState<BitRegister> fockState0(model.getBasisSize());
	FockState<BitRegister> fockState1(model.getBasisSize());
	FockState<BitRegister> fockState2(model.getBasisSize());
	fockState0.getBitRegister().setBit(0, 1);
	fockState1.getBitRegister().setBit(2, 1);
	fockState2.getBitRegister().setBit(0, 1);
	fockState2.getBitRegister().setBit(1, 1);
	fockState2.getBitRegister().setBit(2, 1);

	SumRule sumRule({{0}, {1}}, 1);

	EXPECT_TRUE(sumRule.isSatisfied(fockSpace, fockState0));
	EXPECT_FALSE(sumRule.isSatisfied(fockSpace, fockState1));
	EXPECT_FALSE(sumRule.isSatisfied(fockSpace, fockState2));
}

//TBTKFeature FockStateRule.SumRule.isSatisfied.2 2019-11-05
TEST(SumRule, isSatisfied2){
	Model model;
	for(unsigned int n = 0; n < 4; n++)
		model << HoppingAmplitude(1, {n}, {n});
	model.construct();

	FockSpace<ExtensiveBitRegister> fockSpace(
		&model.getHoppingAmplitudeSet(),
		Statistics::FermiDirac,
		1
	);

	FockState<ExtensiveBitRegister> fockState0(model.getBasisSize());
	FockState<ExtensiveBitRegister> fockState1(model.getBasisSize());
	FockState<ExtensiveBitRegister> fockState2(model.getBasisSize());
	fockState0.getBitRegister().setBit(0, 1);
	fockState1.getBitRegister().setBit(2, 1);
	fockState2.getBitRegister().setBit(0, 1);
	fockState2.getBitRegister().setBit(1, 1);
	fockState2.getBitRegister().setBit(2, 1);

	SumRule sumRule({{0}, {1}}, 1);

	EXPECT_TRUE(sumRule.isSatisfied(fockSpace, fockState0));
	EXPECT_FALSE(sumRule.isSatisfied(fockSpace, fockState1));
	EXPECT_FALSE(sumRule.isSatisfied(fockSpace, fockState2));
}

//TBTKFeature FockStateRule.SumRule.operatorEqual.1 2019-11-05
TEST(SumRule, operatorEqual1){
	EXPECT_TRUE(SumRule({{0}, {1}}, 1) == SumRule({{0}, {1}}, 1));
}

//TBTKFeature FockStateRule.SumRule.operatorEqual.2 2019-11-05
TEST(SumRule, operatorEqual2){
	EXPECT_FALSE(SumRule({{0}}, 1) == SumRule({{0}, {1}}, 1));
}

//TBTKFeature FockStateRule.SumRule.operatorEqual.3 2019-11-05
TEST(SumRule, operatorEqual3){
	EXPECT_FALSE(SumRule({{0}, {1}}, 1) == SumRule({{0}}, 1));
}

//TBTKFeature FockStateRule.SumRule.operatorEqual.4 2019-11-05
TEST(SumRule, operatorEqual4){
	EXPECT_FALSE(SumRule({{0}, {2}}, 1) == SumRule({{0}, {1}}, 1));
}

//TBTKFeature FockStateRule.SumRule.operatorEqual.5 2019-11-05
TEST(SumRule, operatorEqual5){
	EXPECT_FALSE(SumRule({{0}, {2}}, 1) == SumRule({{0}, {1}}, 2));
}

};
};
