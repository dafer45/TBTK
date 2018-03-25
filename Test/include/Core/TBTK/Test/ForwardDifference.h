#include "TBTK/ForwardDifference.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(ForwardDifference, Constructor){
	ForwardDifference forwardDifference0(0, {1, 2, 3});
	EXPECT_EQ(forwardDifference0.getSize(), 2);

	EXPECT_DOUBLE_EQ(real(forwardDifference0[0].getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(forwardDifference0[0].getAmplitude()), 0);
	EXPECT_TRUE(forwardDifference0[0].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(forwardDifference0[0].getFromIndex().equals({1, 2, 3}));
	EXPECT_TRUE(forwardDifference0[1].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(forwardDifference0[1].getFromIndex().equals({2, 2, 3}));

	ForwardDifference forwardDifference1(1, {1, 2, 3});
	EXPECT_EQ(forwardDifference1.getSize(), 2);

	EXPECT_DOUBLE_EQ(real(forwardDifference1[0].getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(forwardDifference1[0].getAmplitude()), 0);
	EXPECT_TRUE(forwardDifference1[0].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(forwardDifference1[0].getFromIndex().equals({1, 2, 3}));
	EXPECT_TRUE(forwardDifference1[1].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(forwardDifference1[1].getFromIndex().equals({1, 3, 3}));

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			ForwardDifference forwardDifference(3, {1, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
