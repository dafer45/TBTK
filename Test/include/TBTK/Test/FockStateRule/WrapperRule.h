#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace FockStateRule{

//TBTKFeature FockStateRule.WrapperRule.clone.1 2019-11-05
TEST(WrapperRule, clone1){
	WrapperRule wrapperRule(SumRule({{0}, {1}}, 1));

	WrapperRule *clone = wrapperRule.clone();
	EXPECT_EQ(*clone, wrapperRule);

	delete clone;
}

//TBTKFeature FockStateRule.WrapperRule.createNewRule.1 2019-11-05
TEST(WrapperRule, createNewRule1){
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

	WrapperRule wrapperRule0(SumRule({{0}, {1}}, 0));
	WrapperRule wrapperRule1(SumRule({{0}, {1}}, 1));

	EXPECT_EQ(
		wrapperRule0.createNewRule(ladderOperator),
		wrapperRule1
	);
}

//TBTKFeature FockStateRule.WrapperRule.createNewRule.2 2019-11-05
TEST(WrapperRule, createNewRule2){
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

	WrapperRule wrapperRule0(SumRule({{0}, {1}}, 0));
	WrapperRule wrapperRule1(SumRule({{0}, {1}}, 1));

	EXPECT_EQ(wrapperRule0.createNewRule(ladderOperator), wrapperRule1);
}

//TBTKFeature FockStateRule.WrapperRule.isSatisfied.1 2019-11-05
TEST(WrapperRule, isSatisfied1){
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

	WrapperRule wrapperRule(SumRule({{0}, {1}}, 1));

	EXPECT_TRUE(wrapperRule.isSatisfied(fockSpace, fockState0));
	EXPECT_FALSE(wrapperRule.isSatisfied(fockSpace, fockState1));
	EXPECT_FALSE(wrapperRule.isSatisfied(fockSpace, fockState2));
}

//TBTKFeature FockStateRule.WrapperRule.isSatisfied.2 2019-11-05
TEST(WrapperRule, isSatisfied2){
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

	WrapperRule wrapperRule(SumRule({{0}, {1}}, 1));

	EXPECT_TRUE(wrapperRule.isSatisfied(fockSpace, fockState0));
	EXPECT_FALSE(wrapperRule.isSatisfied(fockSpace, fockState1));
	EXPECT_FALSE(wrapperRule.isSatisfied(fockSpace, fockState2));
}

//TBTKFeature FockStateRule.WrapperRule.operatorEqual.1 2019-11-05
TEST(WrapperRule, operatorEqual1){
	EXPECT_TRUE(
		WrapperRule(SumRule({{0}, {1}}, 1))
		== WrapperRule(SumRule({{0}, {1}}, 1))
	);
}

//TBTKFeature FockStateRule.WrapperRule.operatorEqual.2 2019-11-05
TEST(WrapperRule, operatorEqual2){
	EXPECT_FALSE(
		WrapperRule(SumRule({{0}, {1}}, 1))
		== WrapperRule(SumRule({{0}, {1}}, 2))
	);
}

};
};
