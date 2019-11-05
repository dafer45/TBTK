#include "TBTK/FockStateRule/DifferenceRule.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace FockStateRule{

//TBTKFeature FockStateRule.DifferenceRule.clone.1 2019-11-05
TEST(DifferenceRule, clone1){
	DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);

	DifferenceRule *clone = differenceRule.clone();
	EXPECT_EQ(*clone, differenceRule);

	delete clone;
}

//TBTKFeature FockStateRule.DifferenceRule.createNewRule.1 2019-11-05
TEST(DifferenceRule, createNewRule1){
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

	DifferenceRule differenceRule0(
		{{0}, {1}},
		{{2}, {3}},
		0
	);
	DifferenceRule differenceRule1(
		{{0}, {1}},
		{{2}, {3}},
		1
	);

	EXPECT_EQ(
		differenceRule0.createNewRule(ladderOperator),
		differenceRule1
	);
}

//TBTKFeature FockStateRule.DifferenceRule.createNewRule.2 2019-11-05
TEST(DifferenceRule, createNewRule2){
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

	DifferenceRule differenceRule0(
		{{0}, {1}},
		{{2}, {3}},
		0
	);
	DifferenceRule differenceRule1(
		{{0}, {1}},
		{{2}, {3}},
		1
	);

	EXPECT_EQ(
		differenceRule0.createNewRule(ladderOperator),
		differenceRule1
	);
}

//TBTKFeature FockStateRule.DifferenceRule.isSatisfied.1 2019-11-05
TEST(DifferenceRule, isSatisfied1){
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

	DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);

	EXPECT_TRUE(differenceRule.isSatisfied(fockSpace, fockState0));
	EXPECT_FALSE(differenceRule.isSatisfied(fockSpace, fockState1));
	EXPECT_TRUE(differenceRule.isSatisfied(fockSpace, fockState2));
}

//TBTKFeature FockStateRule.DifferenceRule.isSatisfied.2 2019-11-05
TEST(DifferenceRule, isSatisfied2){
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

	DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);

	EXPECT_TRUE(differenceRule.isSatisfied(fockSpace, fockState0));
	EXPECT_FALSE(differenceRule.isSatisfied(fockSpace, fockState1));
	EXPECT_TRUE(differenceRule.isSatisfied(fockSpace, fockState2));
}

//TBTKFeature FockStateRule.DifferenceRule.operatorEqual.1 2019-11-05
TEST(DifferenceRule, operatorEqual1){
	EXPECT_TRUE(
		DifferenceRule(
			{{0}, {1}},
			{{2}, {3}},
			1
		) == DifferenceRule(
			{{0}, {1}},
			{{2}, {3}},
			1
		)
	);
}

//TBTKFeature FockStateRule.DifferenceRule.operatorEqual.2 2019-11-05
TEST(DifferenceRule, operatorEqual2){
	EXPECT_FALSE(
		DifferenceRule(
			{{0}, {1}},
			{{2}, {3}},
			1
		) == DifferenceRule(
			{{0}},
			{{2}, {3}},
			1
		)
	);
}

//TBTKFeature FockStateRule.DifferenceRule.operatorEqual.3 2019-11-05
TEST(DifferenceRule, operatorEqual3){
	EXPECT_FALSE(
		DifferenceRule(
			{{0}, {1}},
			{{2}, {3}},
			1
		) == DifferenceRule(
			{{0}, {1}},
			{{2}},
			1
		)
	);
}

//TBTKFeature FockStateRule.DifferenceRule.operatorEqual.4 2019-11-05
TEST(DifferenceRule, operatorEqual4){
	EXPECT_FALSE(
		DifferenceRule(
			{{0}, {2}},
			{{3}},
			1
		) == DifferenceRule(
			{{0}, {1}},
			{{3}},
			1
		)
	);
}

//TBTKFeature FockStateRule.DifferenceRule.operatorEqual.5 2019-11-05
TEST(DifferenceRule, operatorEqual5){
	EXPECT_FALSE(
		DifferenceRule(
			{{0}},
			{{1}, {3}},
			1
		) == DifferenceRule(
			{{0}},
			{{2}, {3}},
			1
		)
	);
}

//TBTKFeature FockStateRule.DifferenceRule.operatorEqual.6 2019-11-05
TEST(DifferenceRule, operatorEqual6){
	EXPECT_FALSE(
		DifferenceRule(
			{{0}, {1}},
			{{2}, {3}},
			1
		) == DifferenceRule(
			{{0}, {1}},
			{{2}, {3}},
			0
		)
	);
}

};
};
