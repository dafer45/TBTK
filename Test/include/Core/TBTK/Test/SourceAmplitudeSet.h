#include "TBTK/SourceAmplitudeSet.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(SourceAmplitudeSet, Constructor){
	//Not testable on its own.
}

TEST(SourceAmplitudeSet, SerializeToJSON){
	SourceAmplitudeSet sourceAmplitudeSet0;
	sourceAmplitudeSet0.add(SourceAmplitude(1, {1, 2, 3}));
	sourceAmplitudeSet0.add(SourceAmplitude(2, {1, 2, 4}));
	sourceAmplitudeSet0.add(SourceAmplitude(3, {1, 2, 3}));

	SourceAmplitudeSet sourceAmplitudeSet1(
		sourceAmplitudeSet0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	//Check that all added amplitudes are possible to get.
	const std::vector<SourceAmplitude> &sourceAmplitudes0 = sourceAmplitudeSet1.get({1, 2, 3});
	EXPECT_EQ(sourceAmplitudes0.size(), 2);
	EXPECT_DOUBLE_EQ(real(sourceAmplitudes0[0].getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitudes0[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(sourceAmplitudes0[1].getAmplitude()), 3);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitudes0[1].getAmplitude()), 0);
	EXPECT_TRUE(sourceAmplitudes0[0].getIndex().equals({1, 2, 3}));
	EXPECT_TRUE(sourceAmplitudes0[1].getIndex().equals({1, 2, 3}));

	const std::vector<SourceAmplitude> &sourceAmplitudes1 = sourceAmplitudeSet1.get({1, 2, 4});
	EXPECT_EQ(sourceAmplitudes1.size(), 1);
	EXPECT_DOUBLE_EQ(real(sourceAmplitudes1[0].getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitudes1[0].getAmplitude()), 0);
	EXPECT_TRUE(sourceAmplitudes1[0].getIndex().equals({1, 2, 4}));
}

TEST(SourceAmplitudeSet, add){
	SourceAmplitudeSet sourceAmplitudeSet;
	sourceAmplitudeSet.add(SourceAmplitude(1, {1, 2, 3}));
	sourceAmplitudeSet.add(SourceAmplitude(2, {1, 2, 4}));
	sourceAmplitudeSet.add(SourceAmplitude(3, {1, 2, 3}));

	//Fail to add SourceAmplitude with conflicting Index structure (shorter).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			sourceAmplitudeSet.add(SourceAmplitude(0, {1, 2}));
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail to add SourceAmplitude with conflicting Index structure (longer).
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			sourceAmplitudeSet.add(SourceAmplitude(0, {1, 2}));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(SourceAmplitudeSet, get){
	SourceAmplitudeSet sourceAmplitudeSet;
	sourceAmplitudeSet.add(SourceAmplitude(1, {1, 2, 3}));
	sourceAmplitudeSet.add(SourceAmplitude(2, {1, 2, 4}));
	sourceAmplitudeSet.add(SourceAmplitude(3, {1, 2, 3}));

	//Check that all added aplitudes are possible to get.
	const std::vector<SourceAmplitude> &sourceAmplitudes0 = sourceAmplitudeSet.get({1, 2, 3});
	EXPECT_EQ(sourceAmplitudes0.size(), 2);
	EXPECT_DOUBLE_EQ(real(sourceAmplitudes0[0].getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitudes0[0].getAmplitude()), 0);
	EXPECT_DOUBLE_EQ(real(sourceAmplitudes0[1].getAmplitude()), 3);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitudes0[1].getAmplitude()), 0);
	EXPECT_TRUE(sourceAmplitudes0[0].getIndex().equals({1, 2, 3}));
	EXPECT_TRUE(sourceAmplitudes0[1].getIndex().equals({1, 2, 3}));

	const std::vector<SourceAmplitude> &sourceAmplitudes1 = sourceAmplitudeSet.get({1, 2, 4});
	EXPECT_EQ(sourceAmplitudes1.size(), 1);
	EXPECT_DOUBLE_EQ(real(sourceAmplitudes1[0].getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag(sourceAmplitudes1[0].getAmplitude()), 0);
	EXPECT_TRUE(sourceAmplitudes1[0].getIndex().equals({1, 2, 4}));

	//Throw ElementNotFOundException for non-existing elements.
	EXPECT_THROW(sourceAmplitudeSet.get({1, 2, 5}), ElementNotFoundException);

	//Fail for invalid Index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			sourceAmplitudeSet.get({1, -1, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(SourceAmplitudeSet, getSizeInBytes){
	SourceAmplitudeSet sourceAmplitudeSet;
	EXPECT_TRUE(sourceAmplitudeSet.getSizeInBytes() > 0);
}

TEST(SourceAmplitudeSet, Iterator){
	SourceAmplitudeSet sourceAmplitudeSet;
	sourceAmplitudeSet.add(SourceAmplitude(1, {1, 2, 3}));
	sourceAmplitudeSet.add(SourceAmplitude(2, {1, 2, 4}));
	sourceAmplitudeSet.add(SourceAmplitude(3, {1, 2, 3}));

	SourceAmplitudeSet::Iterator iterator = sourceAmplitudeSet.begin();
	EXPECT_FALSE(iterator == sourceAmplitudeSet.end());
	EXPECT_TRUE(iterator != sourceAmplitudeSet.end());
	EXPECT_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2, 3}));

	++iterator;
	EXPECT_FALSE(iterator == sourceAmplitudeSet.end());
	EXPECT_TRUE(iterator != sourceAmplitudeSet.end());
	EXPECT_EQ(real((*iterator).getAmplitude()), 3);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2, 3}));

	++iterator;
	EXPECT_FALSE(iterator == sourceAmplitudeSet.end());
	EXPECT_TRUE(iterator != sourceAmplitudeSet.end());
	EXPECT_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2, 4}));

	++iterator;
	EXPECT_TRUE(iterator == sourceAmplitudeSet.end());
	EXPECT_FALSE(iterator != sourceAmplitudeSet.end());
}

TEST(SourceAmplitudeSet, ConstIterator){
	SourceAmplitudeSet sourceAmplitudeSet;
	sourceAmplitudeSet.add(SourceAmplitude(1, {1, 2, 3}));
	sourceAmplitudeSet.add(SourceAmplitude(2, {1, 2, 4}));
	sourceAmplitudeSet.add(SourceAmplitude(3, {1, 2, 3}));

	SourceAmplitudeSet::ConstIterator iterator = sourceAmplitudeSet.cbegin();
	EXPECT_FALSE(iterator == sourceAmplitudeSet.cend());
	EXPECT_TRUE(iterator != sourceAmplitudeSet.cend());
	EXPECT_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2, 3}));

	++iterator;
	EXPECT_FALSE(iterator == sourceAmplitudeSet.cend());
	EXPECT_TRUE(iterator != sourceAmplitudeSet.cend());
	EXPECT_EQ(real((*iterator).getAmplitude()), 3);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2, 3}));

	++iterator;
	EXPECT_FALSE(iterator == sourceAmplitudeSet.cend());
	EXPECT_TRUE(iterator != sourceAmplitudeSet.cend());
	EXPECT_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_EQ(imag((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getIndex().equals({1, 2, 4}));

	++iterator;
	EXPECT_TRUE(iterator == sourceAmplitudeSet.cend());
	EXPECT_FALSE(iterator != sourceAmplitudeSet.cend());

	//Verify that begin() and end() return ConstIterator for const
	//SourceAmplitudeSet.
	iterator = const_cast<const SourceAmplitudeSet&>(sourceAmplitudeSet).begin();
	iterator = const_cast<const SourceAmplitudeSet&>(sourceAmplitudeSet).end();
}

};
