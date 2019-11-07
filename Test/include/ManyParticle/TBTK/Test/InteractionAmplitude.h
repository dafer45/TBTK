#include "TBTK/InteractionAmplitude.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

std::complex<double> amplitudeCallback(
	const std::vector<Index> &creationOperators,
	const std::vector<Index> &annihilationOperators
){
	return std::complex<double>(3, 4);
}

class InteractionAmplitudeTest : public ::testing::Test{
protected:
	InteractionAmplitude interactionAmplitude[2];

	void SetUp() override{
		interactionAmplitude[0] = InteractionAmplitude(
			std::complex<double>(1, 2),
			{{0}, {1}}, {{2}, {3}}
		);
		interactionAmplitude[1] = InteractionAmplitude(
			amplitudeCallback,
			{{0}, {1}}, {{2}, {3}}
		);
	}
};

//TBTKFeature ManyParticle.InteractionAmplitude.getAmplitude.1 2019-11-07
TEST_F(InteractionAmplitudeTest, getAmplitude1){
	EXPECT_EQ(
		interactionAmplitude[0].getAmplitude(),
		std::complex<double>(1, 2)
	);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getAmplitude.2 2019-11-07
TEST_F(InteractionAmplitudeTest, getAmplitude2){
	EXPECT_EQ(
		interactionAmplitude[1].getAmplitude(),
		std::complex<double>(3, 4)
	);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getNumCreationOperators.1 2019-11-07
TEST_F(InteractionAmplitudeTest, getNumCreationOperators1){
	EXPECT_EQ(interactionAmplitude[0].getNumCreationOperators(), 2);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getNumCreationOperators.2 2019-11-07
TEST_F(InteractionAmplitudeTest, getNum2){
	EXPECT_EQ(interactionAmplitude[1].getNumCreationOperators(), 2);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getNumAnnihilationOperators.1 2019-11-07
TEST_F(InteractionAmplitudeTest, getAnnihilation1){
	EXPECT_EQ(interactionAmplitude[0].getNumAnnihilationOperators(), 2);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getNumAnnihilationOperators.2 2019-11-07
TEST_F(InteractionAmplitudeTest, getAnnihilation2){
	EXPECT_EQ(interactionAmplitude[1].getNumAnnihilationOperators(), 2);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getCreationOperatorIndex.1 2019-11-07
TEST_F(InteractionAmplitudeTest, getCreationOperatorIndex1){
	EXPECT_TRUE(
		interactionAmplitude[0].getCreationOperatorIndex(0).equals({0})
	);
	EXPECT_TRUE(
		interactionAmplitude[0].getCreationOperatorIndex(1).equals({1})
	);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getCreationOperatorIndex.2 2019-11-07
TEST_F(InteractionAmplitudeTest, getCreationOperatorIndex2){
	EXPECT_TRUE(
		interactionAmplitude[1].getCreationOperatorIndex(0).equals({0})
	);
	EXPECT_TRUE(
		interactionAmplitude[1].getCreationOperatorIndex(1).equals({1})
	);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getAnnihilationOperatorIndex.1 2019-11-07
TEST_F(InteractionAmplitudeTest, getAnnihilationOperatorIndex1){
	EXPECT_TRUE(
		interactionAmplitude[0].getAnnihilationOperatorIndex(0).equals({2})
	);
	EXPECT_TRUE(
		interactionAmplitude[0].getAnnihilationOperatorIndex(1).equals({3})
	);
}

//TBTKFeature ManyParticle.InteractionAmplitude.getAnnihilationOperatorIndex.2 2019-11-07
TEST_F(InteractionAmplitudeTest, getAnnihilationOperatorIndex2){
	EXPECT_TRUE(
		interactionAmplitude[1].getAnnihilationOperatorIndex(0).equals({2})
	);
	EXPECT_TRUE(
		interactionAmplitude[1].getAnnihilationOperatorIndex(1).equals({3})
	);
}

};
