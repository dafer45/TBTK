#include "TBTK/FockStateRule/DifferenceRule.h"
#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/FockSpace.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

class FockSpaceTest : public ::testing::Test{
protected:
	Model model;
	FockSpace<BitRegister> fockSpaceFermion;
	FockSpace<BitRegister> fockSpaceBoson;

	void SetUp() override{
		for(unsigned int n = 0; n < 4; n++)
			model << HoppingAmplitude(1, {n}, {n});
		model.construct();

		fockSpaceFermion = FockSpace<BitRegister>(
			&model.getHoppingAmplitudeSet(),
			Statistics::FermiDirac,
			1
		);
		fockSpaceBoson = FockSpace<BitRegister>(
			&model.getHoppingAmplitudeSet(),
			Statistics::BoseEinstein,
			3
		);
	}
};

//TBTKFeature ManyParticle.FockSpace.copy.1 2019-11-06
TEST_F(FockSpaceTest, copy1){
	FockSpace<BitRegister> copy = fockSpaceFermion;
	EXPECT_EQ(
		copy.getHoppingAmplitudeSet(),
		fockSpaceFermion.getHoppingAmplitudeSet()
	);
	LadderOperator<BitRegister> const* const* operatorsCopy
		= fockSpaceFermion.getOperators();
	LadderOperator<BitRegister> const* const* operatorsOriginal
		= fockSpaceFermion.getOperators();
	for(int n = 0; n < model.getBasisSize(); n++){
		for(unsigned int c = 0; c < 2; c++){
			EXPECT_EQ(
				operatorsCopy[n][c].getType(),
				operatorsOriginal[n][c].getType()
			);
			EXPECT_TRUE(
				operatorsCopy[n][c].getPhysicalIndex().equals(
					operatorsOriginal[n][c].getPhysicalIndex()
				)
			);
		}
	}
	EXPECT_EQ(
		copy.getVacuumState().getBitRegister(),
		fockSpaceFermion.getVacuumState().getBitRegister()
	);
}

//TBTKFeature ManyParticle.FockSpace.operatorAssignment.1 2019-11-06
TEST_F(FockSpaceTest, operatorAssignment1){
	FockSpace<BitRegister> copy;
	copy = fockSpaceFermion;
	EXPECT_EQ(
		copy.getHoppingAmplitudeSet(),
		fockSpaceFermion.getHoppingAmplitudeSet()
	);
	LadderOperator<BitRegister> const* const* operatorsCopy
		= fockSpaceFermion.getOperators();
	LadderOperator<BitRegister> const* const* operatorsOriginal
		= fockSpaceFermion.getOperators();
	for(int n = 0; n < model.getBasisSize(); n++){
		for(unsigned int c = 0; c < 2; c++){
			EXPECT_EQ(
				operatorsCopy[n][c].getType(),
				operatorsOriginal[n][c].getType()
			);
			EXPECT_TRUE(
				operatorsCopy[n][c].getPhysicalIndex().equals(
					operatorsOriginal[n][c].getPhysicalIndex()
				)
			);
		}
	}
	EXPECT_EQ(
		copy.getVacuumState().getBitRegister(),
		fockSpaceFermion.getVacuumState().getBitRegister()
	);
}

//TBTKFeature ManyParticle.FockSpace.getOperators.1 2019-11-06
TEST_F(FockSpaceTest, getOperators1){
	LadderOperator<BitRegister> const* const* operators
		= fockSpaceFermion.getOperators();
	for(int n = 0; n < model.getBasisSize(); n++){
		EXPECT_EQ(
			operators[n][0].getType(),
			LadderOperator<BitRegister>::Type::Creation
		);
		EXPECT_EQ(
			operators[n][1].getType(),
			LadderOperator<BitRegister>::Type::Annihilation
		);
		EXPECT_TRUE(operators[n][0].getPhysicalIndex().equals({n}));
		EXPECT_TRUE(operators[n][1].getPhysicalIndex().equals({n}));
	}
}

//TBTKFeature ManyParticle.FockSpace.getVacuumState.1 2019-11-06
TEST_F(FockSpaceTest, getVacuumState1){
	FockState<BitRegister> fockState = fockSpaceFermion.getVacuumState();
	const BitRegister &bitRegister = fockState.getBitRegister();
	for(unsigned int n = 0; n < bitRegister.getNumBits(); n++)
		EXPECT_FALSE(bitRegister.getBit(n));
}

//TBTKFeature ManyParticle.FockSpace.getNumFermions.1 2019-11-06
TEST_F(FockSpaceTest, getNumFermions1){
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize());
	LadderOperator<BitRegister> ladderOperator0(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);
	LadderOperator<BitRegister> ladderOperator1(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		2,
		1,
		1,
		templateState,
		fermionMask
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	ladderOperator0*fockState;
	ladderOperator1*fockState;

	EXPECT_EQ(fockSpaceFermion.getNumFermions(fockState), 2);
}

//TBTKFeature ManyParticle.FockSpace.getNumFermions.2 2019-11-06
TEST_F(FockSpaceTest, getNumFermions2){
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize());
	LadderOperator<BitRegister> ladderOperator0(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::BoseEinstein,
		&model.getHoppingAmplitudeSet(),
		0,
		2,
		3,
		templateState,
		fermionMask
	);
	LadderOperator<BitRegister> ladderOperator1(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::BoseEinstein,
		&model.getHoppingAmplitudeSet(),
		2,
		2,
		3,
		templateState,
		fermionMask
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	ladderOperator0*fockState;
	ladderOperator1*fockState;

	EXPECT_EQ(fockSpaceBoson.getNumFermions(fockState), 0);
}

//TBTKFeature ManyParticle.FockSpace.getNumParticles.1 2019-11-06
TEST_F(FockSpaceTest, getNumParticles1){
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize());
	LadderOperator<BitRegister> ladderOperator0(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);
	LadderOperator<BitRegister> ladderOperator1(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		2,
		1,
		1,
		templateState,
		fermionMask
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	ladderOperator0*fockState;
	ladderOperator1*fockState;

	EXPECT_EQ(fockSpaceFermion.getNumParticles(fockState, {0}), 1);
	EXPECT_EQ(fockSpaceFermion.getNumParticles(fockState, {1}), 0);
	EXPECT_EQ(fockSpaceFermion.getNumParticles(fockState, {2}), 1);
	EXPECT_EQ(fockSpaceFermion.getNumParticles(fockState, {3}), 0);
}

//TBTKFeature ManyParticle.FockSpace.getNumParticles.2 2019-11-06
TEST_F(FockSpaceTest, getNumParticles2){
	FockState<BitRegister> templateState(2*model.getBasisSize());
	BitRegister fermionMask(2*model.getBasisSize());
	LadderOperator<BitRegister> ladderOperator0(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::BoseEinstein,
		&model.getHoppingAmplitudeSet(),
		0,
		2,
		3,
		templateState,
		fermionMask
	);
	LadderOperator<BitRegister> ladderOperator1(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::BoseEinstein,
		&model.getHoppingAmplitudeSet(),
		2,
		2,
		3,
		templateState,
		fermionMask
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	ladderOperator0*fockState;
	ladderOperator0*fockState;
	ladderOperator1*fockState;
	ladderOperator1*fockState;
	ladderOperator1*fockState;

	EXPECT_EQ(fockSpaceBoson.getNumParticles(fockState, {0}), 2);
	EXPECT_EQ(fockSpaceBoson.getNumParticles(fockState, {1}), 0);
	EXPECT_EQ(fockSpaceBoson.getNumParticles(fockState, {2}), 3);
	EXPECT_EQ(fockSpaceBoson.getNumParticles(fockState, {3}), 0);
}

//TBTKFeature ManyParticle.FockSpace.getSumParticles.1 2019-11-06
TEST_F(FockSpaceTest, getSumParticles1){
	FockState<BitRegister> templateState(model.getBasisSize());
	BitRegister fermionMask(model.getBasisSize());
	LadderOperator<BitRegister> ladderOperator0(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		0,
		1,
		1,
		templateState,
		fermionMask
	);
	LadderOperator<BitRegister> ladderOperator1(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::FermiDirac,
		&model.getHoppingAmplitudeSet(),
		2,
		1,
		1,
		templateState,
		fermionMask
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	ladderOperator0*fockState;
	ladderOperator1*fockState;

	EXPECT_EQ(fockSpaceFermion.getSumParticles(fockState, {IDX_ALL}), 2);
	EXPECT_EQ(fockSpaceFermion.getSumParticles(fockState, {0}), 1);
}

//TBTKFeature ManyParticle.FockSpace.getSumParticles.2 2019-11-06
TEST_F(FockSpaceTest, getSumParticles2){
	FockState<BitRegister> templateState(2*model.getBasisSize());
	BitRegister fermionMask(2*model.getBasisSize());
	LadderOperator<BitRegister> ladderOperator0(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::BoseEinstein,
		&model.getHoppingAmplitudeSet(),
		0,
		2,
		3,
		templateState,
		fermionMask
	);
	LadderOperator<BitRegister> ladderOperator1(
		LadderOperator<BitRegister>::Type::Creation,
		Statistics::BoseEinstein,
		&model.getHoppingAmplitudeSet(),
		2,
		2,
		3,
		templateState,
		fermionMask
	);

	FockState<BitRegister> fockState(model.getBasisSize());
	ladderOperator0*fockState;
	ladderOperator0*fockState;
	ladderOperator1*fockState;
	ladderOperator1*fockState;
	ladderOperator1*fockState;

	EXPECT_EQ(fockSpaceBoson.getSumParticles(fockState, {IDX_ALL}), 5);
	EXPECT_EQ(fockSpaceBoson.getSumParticles(fockState, {0}), 2);
}

//TBTKFeature ManyParticle.FockSpace.createFockStateMap.1 2019-11-06
TEST_F(FockSpaceTest, createFockStateMap1){
	FockStateMap::FockStateMap<BitRegister> *fockStateMap
		= fockSpaceFermion.createFockStateMap(-1);

	EXPECT_EQ(fockStateMap->getBasisSize(), pow(2, model.getBasisSize()));

	delete fockStateMap;
}

//TBTKFeature ManyParticle.FockSpace.createFockStateMap.2 2019-11-06
TEST_F(FockSpaceTest, createFockStateMap2){
	FockStateMap::FockStateMap<BitRegister> *fockStateMap
		= fockSpaceFermion.createFockStateMap(2);

	const unsigned int FOUR_CHOSE_TWO = 6;
	EXPECT_EQ(fockStateMap->getBasisSize(), FOUR_CHOSE_TWO);
	for(unsigned int n = 0; n < FOUR_CHOSE_TWO; n++){
		EXPECT_EQ(
			fockSpaceFermion.getSumParticles(
				fockStateMap->getFockState(n),
				{IDX_ALL}
			),
			2
		);
	}

	delete fockStateMap;
}

//TBTKFeature ManyParticle.FockSpace.createFockStateMap.3 2019-11-06
TEST_F(FockSpaceTest, createFockStateMap3){
	FockStateRule::DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);

	FockStateMap::FockStateMap<BitRegister> *fockStateMap0
		= fockSpaceFermion.createFockStateMap(differenceRule);

	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(differenceRule);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap1
		= fockSpaceFermion.createFockStateMap(fockStateRuleSet);

	EXPECT_EQ(fockStateMap0->getBasisSize(), fockStateMap1->getBasisSize());
	for(unsigned int n = 0; n < fockStateMap0->getBasisSize(); n++){
		EXPECT_EQ(
			fockStateMap0->getFockState(n).getBitRegister(),
			fockStateMap1->getFockState(n).getBitRegister()
		);
	}
}

//TBTKFeature ManyParticle.FockSpace.createFockStateMap.4 2019-11-06
TEST_F(FockSpaceTest, createFockStateMap4){
	FockStateRule::DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);
	FockStateRule::SumRule sumRule(
		{{0}, {1}, {2}, {3}},
		3
	);

	FockStateMap::FockStateMap<BitRegister> *fockStateMap0
		= fockSpaceFermion.createFockStateMap({
			FockStateRule::WrapperRule(differenceRule),
			FockStateRule::WrapperRule(sumRule),
		});

	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(differenceRule);
	fockStateRuleSet.addFockStateRule(sumRule);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap1
		= fockSpaceFermion.createFockStateMap(fockStateRuleSet);

	EXPECT_EQ(fockStateMap0->getBasisSize(), fockStateMap1->getBasisSize());
	for(unsigned int n = 0; n < fockStateMap0->getBasisSize(); n++){
		EXPECT_EQ(
			fockStateMap0->getFockState(n).getBitRegister(),
			fockStateMap1->getFockState(n).getBitRegister()
		);
	}
}

//TBTKFeature ManyParticle.FockSpace.createFockStateMap.5 2019-11-06
TEST_F(FockSpaceTest, createFockStateMap5){
	FockStateRule::DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);
	FockStateRule::SumRule sumRule(
		{{0}, {1}, {2}, {3}},
		3
	);

	std::vector<FockStateRule::WrapperRule> rules = {
		FockStateRule::WrapperRule(differenceRule),
		FockStateRule::WrapperRule(sumRule),
	};
	FockStateMap::FockStateMap<BitRegister> *fockStateMap0
		= fockSpaceFermion.createFockStateMap(rules);

	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(differenceRule);
	fockStateRuleSet.addFockStateRule(sumRule);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap1
		= fockSpaceFermion.createFockStateMap(fockStateRuleSet);

	EXPECT_EQ(fockStateMap0->getBasisSize(), fockStateMap1->getBasisSize());
	for(unsigned int n = 0; n < fockStateMap0->getBasisSize(); n++){
		EXPECT_EQ(
			fockStateMap0->getFockState(n).getBitRegister(),
			fockStateMap1->getFockState(n).getBitRegister()
		);
	}
}

//TBTKFeature ManyParticle.FockSpace.createFockStateMap.6 2019-11-06
TEST_F(FockSpaceTest, createFockStateMap6){
	FockStateRule::DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);
	FockStateRule::SumRule sumRule(
		{{0}, {1}, {2}, {3}},
		3
	);

	FockStateRuleSet fockStateRuleSet;
	fockStateRuleSet.addFockStateRule(differenceRule);
	fockStateRuleSet.addFockStateRule(sumRule);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap
		= fockSpaceFermion.createFockStateMap(fockStateRuleSet);

	BitRegister expectedResult[2];
	for(unsigned int n = 0; n < 2; n++){
		for(
			unsigned int c = 0;
			c < expectedResult[n].getNumBits();
			c++
		){
			expectedResult[n].setBit(c, 0);
		}
	}
	expectedResult[0].setBit(0, 1);
	expectedResult[0].setBit(1, 1);
	expectedResult[0].setBit(2, 1);
	expectedResult[1].setBit(0, 1);
	expectedResult[1].setBit(1, 1);
	expectedResult[1].setBit(3, 1);

	EXPECT_EQ(fockStateMap->getBasisSize(), 2);
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		EXPECT_EQ(
			fockStateMap->getFockState(n).getBitRegister(),
			expectedResult[n]
		);
	}
}

};
