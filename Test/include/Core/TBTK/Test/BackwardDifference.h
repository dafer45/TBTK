#include "TBTK/BackwardDifference.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(BackwardDifference, Constructor){
	BackwardDifference backwardDifference0(0, {1, 2, 3});
	EXPECT_EQ(backwardDifference0.getSize(), 2);

	EXPECT_DOUBLE_EQ(real(backwardDifference0[0].getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(backwardDifference0[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(backwardDifference0[1].getAmplitude()), -1);
	EXPECT_DOUBLE_EQ(imag(backwardDifference0[1].getAmplitude()), 0);
	EXPECT_TRUE(backwardDifference0[0].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(backwardDifference0[0].getFromIndex().equals({0, 2, 3}));
	EXPECT_TRUE(backwardDifference0[1].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(backwardDifference0[1].getFromIndex().equals({1, 2, 3}));

	BackwardDifference backwardDifference1(1, {1, 2, 3});
	EXPECT_EQ(backwardDifference1.getSize(), 2);

	EXPECT_DOUBLE_EQ(real(backwardDifference1[0].getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(backwardDifference1[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(backwardDifference1[1].getAmplitude()), -1);
	EXPECT_DOUBLE_EQ(imag(backwardDifference1[1].getAmplitude()), 0);
	EXPECT_TRUE(backwardDifference1[0].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(backwardDifference1[0].getFromIndex().equals({1, 1, 3}));
	EXPECT_TRUE(backwardDifference1[1].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(backwardDifference1[1].getFromIndex().equals({1, 2, 3}));

	BackwardDifference backwardDifference2(0, {1, 2, 3}, 0.1);
	EXPECT_DOUBLE_EQ(real(backwardDifference2[0].getAmplitude()), 10);
	EXPECT_DOUBLE_EQ(imag(backwardDifference2[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(backwardDifference2[1].getAmplitude()), -10);
	EXPECT_DOUBLE_EQ(imag(backwardDifference2[1].getAmplitude()), 0);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			BackwardDifference backwardDifference(3, {1, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			BackwardDifference backwardDifference(0, {0, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
