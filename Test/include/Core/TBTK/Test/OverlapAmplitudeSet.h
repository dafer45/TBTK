#include "TBTK/OverlapAmplitudeSet.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(OverlapAmplitudeSet, Constructor){
	//Not testable on its own.
}

TEST(OverlapAmplitudeSet, SerializeToJSON){
	OverlapAmplitudeSet overlapAmplitudeSet0;
	overlapAmplitudeSet0.add(OverlapAmplitude(1, {1, 2, 3}, {4, 5, 6}));
	overlapAmplitudeSet0.add(OverlapAmplitude(2, {1, 2, 4}, {4, 5, 7}));

	OverlapAmplitudeSet overlapAmplitudeSet1(
		overlapAmplitudeSet0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	//Check that all added amplitudes are possible to get.
	EXPECT_DOUBLE_EQ(
		real(
			overlapAmplitudeSet1.get(
				{1, 2, 3},
				{4, 5, 6}
			).getAmplitude()
		),
		1
	);
	EXPECT_DOUBLE_EQ(
		imag(
			overlapAmplitudeSet1.get(
				{1, 2, 3},
				{4, 5, 6}
			).getAmplitude()
		),
		0
	);
	EXPECT_DOUBLE_EQ(
		real(
			overlapAmplitudeSet1.get(
				{1, 2, 4},
				{4, 5, 7}
			).getAmplitude()
		),
		2
	);
	EXPECT_DOUBLE_EQ(
		imag(
			overlapAmplitudeSet1.get(
				{1, 2, 4},
				{4, 5, 7}
			).getAmplitude()
		),
		0
	);
	EXPECT_TRUE(
		overlapAmplitudeSet1.get(
			{1, 2, 3},
			{4, 5, 6}
		).getBraIndex().equals({1, 2, 3})
	);
	EXPECT_TRUE(
		overlapAmplitudeSet1.get(
			{1, 2, 3},
			{4, 5, 6}
		).getKetIndex().equals({4, 5, 6})
	);
	EXPECT_TRUE(
		overlapAmplitudeSet1.get(
			{1, 2, 4},
			{4, 5, 7}
		).getBraIndex().equals({1, 2, 4})
	);
	EXPECT_TRUE(
		overlapAmplitudeSet1.get(
			{1, 2, 4},
			{4, 5, 7}
		).getKetIndex().equals({4, 5, 7})
	);

	EXPECT_FALSE(overlapAmplitudeSet1.getAssumeOrthonormalBasis());
	EXPECT_TRUE(
		OverlapAmplitudeSet(
			OverlapAmplitudeSet().serialize(Serializable::Mode::JSON),
			Serializable::Mode::JSON
		).getAssumeOrthonormalBasis()
	);
}

TEST(OverlapAmplitudeSet, add){
	OverlapAmplitudeSet overlapAmplitudeSet;
	overlapAmplitudeSet.add(OverlapAmplitude(1, {1, 2, 3}, {4, 5, 6}));
	overlapAmplitudeSet.add(OverlapAmplitude(2, {1, 2, 4}, {4, 5, 7}));

	//Fail to add OverlapAmplitude with conflicting Index structure
	//(shorter).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			overlapAmplitudeSet.add(
				OverlapAmplitude(0, {1, 2}, {2, 3, 4})
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add OverlapAmplitude with conflicting Index structure
	//(longer).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			overlapAmplitudeSet.add(
				OverlapAmplitude(0, {1, 2, 3, 4}, {2, 3, 4})
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(OverlapAmplitudeSet, get){
	OverlapAmplitudeSet overlapAmplitudeSet;
	overlapAmplitudeSet.add(OverlapAmplitude(1, {1, 2, 3}, {4, 5, 6}));
	overlapAmplitudeSet.add(OverlapAmplitude(2, {1, 2, 4}, {4, 5, 7}));

	//Check that all added aplitudes are possible to get.
	EXPECT_DOUBLE_EQ(
		real(
			overlapAmplitudeSet.get(
				{1, 2, 3},
				{4, 5, 6}
			).getAmplitude()
		),
		1
	);
	EXPECT_DOUBLE_EQ(
		imag(
			overlapAmplitudeSet.get(
				{1, 2, 3},
				{4, 5, 6}
			).getAmplitude()
		),
		0
	);
	EXPECT_DOUBLE_EQ(
		real(
			overlapAmplitudeSet.get(
				{1, 2, 4},
				{4, 5, 7}
			).getAmplitude()
		),
		2
	);
	EXPECT_DOUBLE_EQ(
		imag(
			overlapAmplitudeSet.get(
				{1, 2, 4},
				{4, 5, 7}
			).getAmplitude()
		),
		0
	);
	EXPECT_TRUE(
		overlapAmplitudeSet.get(
			{1, 2, 3},
			{4, 5, 6}
		).getBraIndex().equals({1, 2, 3})
	);
	EXPECT_TRUE(
		overlapAmplitudeSet.get(
			{1, 2, 3},
			{4, 5, 6}
		).getKetIndex().equals({4, 5, 6})
	);
	EXPECT_TRUE(
		overlapAmplitudeSet.get(
			{1, 2, 4},
			{4, 5, 7}
		).getBraIndex().equals({1, 2, 4})
	);
	EXPECT_TRUE(
		overlapAmplitudeSet.get(
			{1, 2, 4},
			{4, 5, 7}
		).getKetIndex().equals({4, 5, 7})
	);

	//Throw ElementNotFoundException for non-existing elements.
	EXPECT_THROW(
		overlapAmplitudeSet.get({1, 2, 5}, {4, 5, 6}),
		ElementNotFoundException
	);

	//Fail for invalid Index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			overlapAmplitudeSet.get({1, -1, 3}, {4, 5, 6});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(OverlapAmplitudeSet, getAssumeOrthonormalBasis){
	OverlapAmplitudeSet overlapAmplitudeSet;
	EXPECT_TRUE(overlapAmplitudeSet.getAssumeOrthonormalBasis());

	overlapAmplitudeSet.add(OverlapAmplitude(1, {1, 2, 3}, {4, 5, 6}));
	EXPECT_FALSE(overlapAmplitudeSet.getAssumeOrthonormalBasis());
}

TEST(OverlapAmplitudeSet, getSizeInBytes){
	OverlapAmplitudeSet overlapAmplitudeSet;
	EXPECT_TRUE(overlapAmplitudeSet.getSizeInBytes() > 0);
}

TEST(OverlapAmplitudeSet, Iterator){
	OverlapAmplitudeSet overlapAmplitudeSet;
	overlapAmplitudeSet.add(OverlapAmplitude(1, {1, 2, 3}, {4, 5, 6}));
	overlapAmplitudeSet.add(OverlapAmplitude(2, {1, 2, 4}, {4, 5, 7}));

	OverlapAmplitudeSet::Iterator iterator = overlapAmplitudeSet.begin();
	EXPECT_FALSE(iterator == overlapAmplitudeSet.end());
	EXPECT_TRUE(iterator != overlapAmplitudeSet.end());
	EXPECT_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getBraIndex().equals({1, 2, 3}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({4, 5, 6}));

	++iterator;
	EXPECT_FALSE(iterator == overlapAmplitudeSet.end());
	EXPECT_TRUE(iterator != overlapAmplitudeSet.end());
	EXPECT_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getBraIndex().equals({1, 2, 4}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({4, 5, 7}));

	++iterator;
	EXPECT_TRUE(iterator == overlapAmplitudeSet.end());
	EXPECT_FALSE(iterator != overlapAmplitudeSet.end());
}

TEST(OverlapAmplitudeSet, ConstIterator){
	OverlapAmplitudeSet overlapAmplitudeSet;
	overlapAmplitudeSet.add(OverlapAmplitude(1, {1, 2, 3}, {4, 5, 6}));
	overlapAmplitudeSet.add(OverlapAmplitude(2, {1, 2, 4}, {4, 5, 7}));

	OverlapAmplitudeSet::ConstIterator iterator
		= overlapAmplitudeSet.cbegin();
	EXPECT_FALSE(iterator == overlapAmplitudeSet.cend());
	EXPECT_TRUE(iterator != overlapAmplitudeSet.cend());
	EXPECT_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getBraIndex().equals({1, 2, 3}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({4, 5, 6}));

	++iterator;
	EXPECT_FALSE(iterator == overlapAmplitudeSet.cend());
	EXPECT_TRUE(iterator != overlapAmplitudeSet.cend());
	EXPECT_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getBraIndex().equals({1, 2, 4}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({4, 5, 7}));

	++iterator;
	EXPECT_TRUE(iterator == overlapAmplitudeSet.cend());
	EXPECT_FALSE(iterator != overlapAmplitudeSet.cend());

	//Verify that begin() and end() return ConstIterator for const
	//OverlapAmplitudeSet.
	iterator = const_cast<const OverlapAmplitudeSet&>(overlapAmplitudeSet).begin();
	iterator = const_cast<const OverlapAmplitudeSet&>(overlapAmplitudeSet).end();
}

};
