#include "TBTK/InteractionAmplitudeSet.h"
#include "TBTK/Model.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

class InteractionAmplitudeSetTest : public ::testing::Test{
protected:
	Model model;
	InteractionAmplitudeSet interactionAmplitudeSet;

	void SetUp() override{
		for(unsigned int n = 0; n < 4; n++)
			model << HoppingAmplitude(1, {n}, {n});
		model.construct();

		interactionAmplitudeSet.addIA(
			InteractionAmplitude(
				1,
				{{0}, {1}},
				{{1}, {0}}
			)
		);
		interactionAmplitudeSet.addIA(
			InteractionAmplitude(
				2,
				{{2}, {3}},
				{{3}, {2}}
			)
		);
	}
};

//TBTKFeature ManyParticle.InteractionAmplitudeSet.addIA.1 2019-11-07
TEST_F(InteractionAmplitudeSetTest, addIA1){
	interactionAmplitudeSet.addIA(
		InteractionAmplitude(
			3,
			{{0}, {1}},
			{{2}, {3}}
		)
	);
	EXPECT_EQ(interactionAmplitudeSet.getNumInteractionAmplitudes(), 3);
}

//TBTKFeature ManyParticle.InteractionAmplitudeSet.getNumInteractionAmplitudes.1 2019-11-07
TEST_F(InteractionAmplitudeSetTest, getNumInteractionAmplitudes1){
	EXPECT_EQ(interactionAmplitudeSet.getNumInteractionAmplitudes(), 2);
}

//TBTKFeature ManyParticle.InteractionAmplitudeSet.getInteractionAmplitude.1 2019-11-07
TEST_F(InteractionAmplitudeSetTest, getInteractionAmplitude1){
	const InteractionAmplitude &interactionAmplitude0
		= interactionAmplitudeSet.getInteractionAmplitude(0);
	EXPECT_EQ(
		interactionAmplitude0.getAmplitude(),
		std::complex<double>(1, 0)
	);
	EXPECT_EQ(interactionAmplitude0.getNumCreationOperators(), 2);
	EXPECT_EQ(interactionAmplitude0.getNumAnnihilationOperators(), 2);
	EXPECT_TRUE(
		interactionAmplitude0.getCreationOperatorIndex(0).equals({0})
	);
	EXPECT_TRUE(
		interactionAmplitude0.getCreationOperatorIndex(1).equals({1})
	);
	EXPECT_TRUE(
		interactionAmplitude0.getAnnihilationOperatorIndex(
			0
		).equals({1})
	);
	EXPECT_TRUE(
		interactionAmplitude0.getAnnihilationOperatorIndex(
			1
		).equals({0})
	);

	const InteractionAmplitude &interactionAmplitude1
		= interactionAmplitudeSet.getInteractionAmplitude(1);
	EXPECT_EQ(
		interactionAmplitude1.getAmplitude(),
		std::complex<double>(2, 0)
	);
	EXPECT_EQ(interactionAmplitude1.getNumCreationOperators(), 2);
	EXPECT_EQ(interactionAmplitude1.getNumAnnihilationOperators(), 2);
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
