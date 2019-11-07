#include "TBTK/FockStateRule/DifferenceRule.h"
#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/ManyParticleContext.h"
#include "TBTK/SingleParticleContext.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

class ManyParticleContextTest : public ::testing::Test{
protected:
	SingleParticleContext singleParticleContext[2];
	ManyParticleContext manyParticleContext[2];

	void SetUp() override{
		HoppingAmplitudeSet &hoppingAmplitudeSet0
			= singleParticleContext[0].getHoppingAmplitudeSet();
		for(unsigned int n = 0; n < 4; n++){
			hoppingAmplitudeSet0.add(
				HoppingAmplitude(1, {n}, {n})
			);
		}
		hoppingAmplitudeSet0.construct();

		HoppingAmplitudeSet &hoppingAmplitudeSet1
			= singleParticleContext[1].getHoppingAmplitudeSet();
		for(unsigned int n = 0; n < 64; n++){
			hoppingAmplitudeSet1.add(
				HoppingAmplitude(1, {n}, {n})
			);
		}
		hoppingAmplitudeSet1.construct();

		for(unsigned int n = 0; n < 2; n++){
			manyParticleContext[n]
				= ManyParticleContext(&singleParticleContext[n]);
		}
	}
};

//TBTKFeature ManyParticle.ManyParticleContext.wrapsBitRegister.1 2019-11-07
TEST_F(ManyParticleContextTest, wrapsBitRegister1){
	EXPECT_TRUE(manyParticleContext[0].wrapsBitRegister());
	EXPECT_FALSE(manyParticleContext[0].wrapsExtensiveBitRegister());
}

//TBTKFeature ManyParticle.ManyParticleContext.wrapsExtensiveBitRegister.1 2019-11-07
TEST_F(ManyParticleContextTest, wrapsExtensiveBitRegister1){
	EXPECT_FALSE(manyParticleContext[1].wrapsBitRegister());
	EXPECT_TRUE(manyParticleContext[1].wrapsExtensiveBitRegister());
}

//TBTKFeature ManyParticle.ManyParticleContext.getFockSpaceBitRegister.1 2019-11-07
TEST_F(ManyParticleContextTest, getFockSpaceBitRegister1){
	const FockSpace<BitRegister> *fockSpaceBitRegister
		= manyParticleContext[0].getFockSpaceBitRegister();

	EXPECT_TRUE(fockSpaceBitRegister != nullptr);
}

//TBTKFeature ManyParticle.ManyParticleContext.getFockSpaceBitRegister.2 2019-11-07
TEST_F(ManyParticleContextTest, getFockSpaceBitRegister2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			manyParticleContext[
				0
			].getFockSpaceExtensiveBitRegister();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature ManyParticle.ManyParticleContext.getFockSpaceExtensiveBitRegister.1 2019-11-07
TEST_F(ManyParticleContextTest, getFockSpaceExtensiveBitRegister1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			manyParticleContext[1].getFockSpaceBitRegister();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature ManyParticle.ManyParticleContext.getFockSpaceBitRegister.2 2019-11-07
TEST_F(ManyParticleContextTest, getFockSpaceExtensiveBitRegister2){
	const FockSpace<ExtensiveBitRegister> *fockSpaceExtensiveBitRegister
		= manyParticleContext[1].getFockSpaceExtensiveBitRegister();

	EXPECT_TRUE(fockSpaceExtensiveBitRegister != nullptr);
}

//TBTKFeature ManyParticle.ManyParticleContext.addFockStateRule.1 2019-11-07
//TBTKFeature ManyParticle.ManyParticleContext.getFockStateRuleSet.1 2019-11-07
TEST_F(ManyParticleContextTest, getFockStateRuleSet1){
	FockStateRule::DifferenceRule differenceRule(
		{{0}, {1}},
		{{2}, {3}},
		1
	);
	FockStateRule::SumRule sumRule({{0}, {1}, {2}, {3}}, 3);

	manyParticleContext[0].addFockStateRule(differenceRule);
	manyParticleContext[0].addFockStateRule(sumRule);

	FockStateRuleSet reference;
	reference.addFockStateRule(differenceRule);
	reference.addFockStateRule(sumRule);

	EXPECT_EQ(manyParticleContext[0].getFockStateRuleSet(), reference);
}

//TBTKFeature ManyParticle.ManyParticleContext.addIA.1 2019-11-07
//TBTKFeature ManyParticle.ManyParticleContext.getInteractionAmplitudeSet.1 2019-11-07
TEST_F(ManyParticleContextTest, getInteractionAmplitudeSet1){
	manyParticleContext[0].addIA(
		InteractionAmplitude(1, {{0}, {1}}, {{1}, {0}})
	);
	manyParticleContext[0].addIA(
		InteractionAmplitude(2, {{2}, {3}}, {{3}, {2}})
	);

	const InteractionAmplitudeSet *interactionAmplitudeSet
		= manyParticleContext[0].getInteractionAmplitudeSet();

	EXPECT_EQ(interactionAmplitudeSet->getNumInteractionAmplitudes(), 2);

	const InteractionAmplitude &interactionAmplitude0
		= interactionAmplitudeSet->getInteractionAmplitude(0);
	EXPECT_EQ(
		interactionAmplitude0.getAmplitude(),
		std::complex<double>(1, 0)
	);
	EXPECT_TRUE(
		interactionAmplitude0.getCreationOperatorIndex(0).equals({0})
	);
	EXPECT_TRUE(
		interactionAmplitude0.getCreationOperatorIndex(1).equals({1})
	);
	EXPECT_TRUE(
		interactionAmplitude0.getAnnihilationOperatorIndex(0).equals(
			{1}
		)
	);
	EXPECT_TRUE(
		interactionAmplitude0.getAnnihilationOperatorIndex(1).equals(
			{0}
		)
	);

	const InteractionAmplitude &interactionAmplitude1
		= interactionAmplitudeSet->getInteractionAmplitude(1);
	EXPECT_EQ(
		interactionAmplitude1.getAmplitude(),
		std::complex<double>(2, 0)
	);
	EXPECT_TRUE(
		interactionAmplitude1.getCreationOperatorIndex(0).equals({2})
	);
	EXPECT_TRUE(
		interactionAmplitude1.getCreationOperatorIndex(1).equals({3})
	);
	EXPECT_TRUE(
		interactionAmplitude1.getAnnihilationOperatorIndex(0).equals(
			{3}
		)
	);
	EXPECT_TRUE(
		interactionAmplitude1.getAnnihilationOperatorIndex(1).equals(
			{2}
		)
	);
}

};
