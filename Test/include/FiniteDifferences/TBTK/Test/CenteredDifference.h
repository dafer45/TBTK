#include "TBTK/CenteredDifference.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(CenteredDifference, Constructor){
	CenteredDifference centeredDifference0(0, {1, 2, 3});
	EXPECT_EQ(centeredDifference0.getSize(), 2);

	EXPECT_DOUBLE_EQ(real(centeredDifference0[0].getAmplitude()), 1/2.);
	EXPECT_DOUBLE_EQ(imag(centeredDifference0[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(centeredDifference0[1].getAmplitude()), -1/2.);
	EXPECT_DOUBLE_EQ(imag(centeredDifference0[1].getAmplitude()), 0);
	EXPECT_TRUE(centeredDifference0[0].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(centeredDifference0[0].getFromIndex().equals({0, 2, 3}));
	EXPECT_TRUE(centeredDifference0[1].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(centeredDifference0[1].getFromIndex().equals({2, 2, 3}));

	CenteredDifference centeredDifference1(1, {1, 2, 3});
	EXPECT_EQ(centeredDifference1.getSize(), 2);

	EXPECT_DOUBLE_EQ(real(centeredDifference1[0].getAmplitude()), 1/2.);
	EXPECT_DOUBLE_EQ(imag(centeredDifference1[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(centeredDifference1[1].getAmplitude()), -1/2.);
	EXPECT_DOUBLE_EQ(imag(centeredDifference1[1].getAmplitude()), 0);
	EXPECT_TRUE(centeredDifference1[0].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(centeredDifference1[0].getFromIndex().equals({1, 1, 3}));
	EXPECT_TRUE(centeredDifference1[1].getToIndex().equals({1, 2, 3}));
	EXPECT_TRUE(centeredDifference1[1].getFromIndex().equals({1, 3, 3}));

	CenteredDifference centeredDifference2(1, {1, 2, 3}, 0.1);
	EXPECT_DOUBLE_EQ(real(centeredDifference2[0].getAmplitude()), 5);
	EXPECT_DOUBLE_EQ(imag(centeredDifference2[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(centeredDifference2[1].getAmplitude()), -5);
	EXPECT_DOUBLE_EQ(imag(centeredDifference2[1].getAmplitude()), 0);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			CenteredDifference centeredDifference(3, {1, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			CenteredDifference centeredDifference(0, {0, 2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
