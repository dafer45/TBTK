#include "TBTK/Subindex.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

//TBTKFeature Core.Subindex.Construction.1 2019-09-19
TEST(Subindex, Constructor0){
	Subindex subindex;
}

//TBTKFeature Core.Subindex.Construction.2.C++ 2019-09-19
TEST(Subindex, Constructor1){
	Subindex subindex0((int)7);
	Subindex subindex1(Integer(7));
	Subindex subindex2((unsigned int)7);

	EXPECT_EQ(subindex0, subindex1);
	EXPECT_EQ(subindex0, subindex2);
}

//TBTKFeature Core.Subindex.Serialization.1 2019-09-22
TEST(Subindex, serializeToJSON){
	Subindex subindex0(7);
	Subindex subindex1(
		subindex0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(subindex0, subindex1);
}

TEST(Subindex, isWildcard){
	Subindex subindex0(IDX_ALL);
	Subindex subindex1(IDX_ALL + 1);
	Subindex subindex2(0);
	Subindex subindex3(1);

	//TBTKFeature Core.Subindex.isWildcard.1 2019-09-21
	EXPECT_TRUE(subindex0.isWildcard());
	//TBTKFeature Core.Subindex.isWildcard.2 2019-09-21
	EXPECT_FALSE(subindex1.isWildcard());
	EXPECT_FALSE(subindex2.isWildcard());
	EXPECT_FALSE(subindex3.isWildcard());
}

TEST(Subindex, isLabeledWildcard){
	Subindex subindex0(IDX_ALL_(0));
	Subindex subindex1(IDX_ALL_(1));
	Subindex subindex2(IDX_ALL);
	Subindex subindex3(0);
	Subindex subindex4(1);

	//TBTKFeature Core.Subindex.isLabeledWildcard.1 2019-09-21
	EXPECT_TRUE(subindex0.isLabeledWildcard());
	EXPECT_TRUE(subindex1.isLabeledWildcard());
	//TBTKFeature Core.Subindex.isLabeledWildcard.2 2019-09-21
	EXPECT_FALSE(subindex2.isLabeledWildcard());
	EXPECT_FALSE(subindex3.isLabeledWildcard());
	EXPECT_FALSE(subindex4.isLabeledWildcard());
}

TEST(Subindex, getWildcardLabel){
	Subindex subindex0(IDX_ALL_(0));
	Subindex subindex1(IDX_ALL_(1));
	Subindex subindex2(1);
	Subindex subindex3(IDX_ALL);

	//TBTKFeature Core.Subindex.getWildcardLabel.1 2019-10-17
	EXPECT_EQ(subindex0.getWildcardLabel(), 0);
	//TBTKFeature Core.Subindex.getWildcardLabel.2 2019-10-17
	EXPECT_EQ(subindex1.getWildcardLabel(), 1);
	//TBTKFeature Core.Subindex.getWildcardLabel.3 2019-10-17
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			subindex2.getWildcardLabel();
		},
		::testing::ExitedWithCode(1),
		""
	);
	//TBTKFeature Core.Subindex.getWildcardLabel.4 2019-10-17
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			subindex3.getWildcardLabel();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Subindex, isSummationIndex){
	Subindex subindex0(IDX_SUM_ALL);
	Subindex subindex1(IDX_SUM_ALL + 1);
	Subindex subindex2(0);
	Subindex subindex3(1);

	//TBTKFeature Core.Subindex.isSummationIndex.1 2019-09-21
	EXPECT_TRUE(subindex0.isSummationIndex());
	//TBTKFeature Core.Subindex.isSummationIndex.2 2019-09-22
	EXPECT_FALSE(subindex1.isSummationIndex());
	EXPECT_FALSE(subindex2.isSummationIndex());
	EXPECT_FALSE(subindex3.isSummationIndex());
}

TEST(Subindex, isRangeIndex){
	Subindex subindex0(IDX_X);
	Subindex subindex1(IDX_Y);
	Subindex subindex2(IDX_ALL);
	Subindex subindex3(0);
	Subindex subindex4(1);

	EXPECT_TRUE(subindex0.isRangeIndex());
	EXPECT_TRUE(subindex1.isRangeIndex());
	EXPECT_FALSE(subindex2.isRangeIndex());
	EXPECT_FALSE(subindex3.isRangeIndex());
	EXPECT_FALSE(subindex4.isRangeIndex());
}

TEST(Subindex, isSpinIndex){
	Subindex subindex0(IDX_SPIN);
	Subindex subindex1(IDX_SPIN + 1);
	Subindex subindex2(0);
	Subindex subindex3(1);

	//TBTKFeature Core.Subindex.isSpinIndex.1 2019-09-22
	EXPECT_TRUE(subindex0.isSpinIndex());
	//TBTKFeature Core.Subindex.isSpinIndex.2 2019-09-22
	EXPECT_FALSE(subindex1.isSpinIndex());
	EXPECT_FALSE(subindex2.isSpinIndex());
	EXPECT_FALSE(subindex3.isSpinIndex());
}

TEST(Subindex, isIndexSeparator){
	Subindex subindex0(IDX_SEPARATOR);
	Subindex subindex1(IDX_SEPARATOR + 1);
	Subindex subindex2(0);
	Subindex subindex3(1);

	//TBTKFeature Core.Subindex.isIndexSeparator.1 2019-09-21
	EXPECT_TRUE(subindex0.isIndexSeparator());
	//TBTKFeature Core.Subindex.isIndexSeparator.2 2019-09-22
	EXPECT_FALSE(subindex1.isIndexSeparator());
	EXPECT_FALSE(subindex2.isIndexSeparator());
	EXPECT_FALSE(subindex3.isIndexSeparator());
}

//TBTKFeature Core.Subindex.operatorInt.1.C++ 2019-09-22
TEST(Subindex, operatorInt){
	Subindex subindex(7);
	int i = (int)subindex;

	EXPECT_EQ(i, 7);
}

//TBTKFeature Core.Subindex.operatorAssignment.1.C++ 2019-09-22
TEST(Subindex, operatorAssignment){
	Subindex subindex0(7);
	Subindex subindex1(0);
	Subindex subindex2(0);
	Subindex subindex3(0);
	subindex1 = Integer(7);
	subindex2 = (int)7;
	subindex3 = (unsigned int)7;

	EXPECT_EQ(subindex0, subindex1);
	EXPECT_EQ(subindex0, subindex2);
	EXPECT_EQ(subindex0, subindex3);
}

//TBTKFeature Core.Subindex.operatorAdditionAssignment.1.C++ 2019-09-22
TEST(Subindex, operatorAdditionAssignment){
	Subindex subindex(7);
	subindex += 1;

	EXPECT_EQ(subindex, 8);
}

//TBTKFeature Core.Subindex.operatorAddition.1.C++ 2019-09-22
TEST(Subindex, operatorAddition){
	Subindex subindex0(7);
	Subindex subindex1(1);

	EXPECT_EQ(subindex0 + subindex1, 8);
}

//TBTKFeature Core.Subindex.operatorSubtractionAssignment.1.C++ 2019-09-22
TEST(Subindex, operatorSubtractionAssignment){
	Subindex subindex(7);
	subindex -= 1;

	EXPECT_EQ(subindex, 6);
}

//TBTKFeature Core.Subindex.operatorSubtraction.1.C++ 2019-09-22
TEST(Subindex, operatorSubtraction){
	Subindex subindex0(7);
	Subindex subindex1(1);

	EXPECT_EQ(subindex0 - subindex1, 6);
}

//TBTKFeature Core.Subindex.operatorMultiplicationAssignment.1.C++ 2019-09-22
TEST(Subindex, operatorMultiplicationAssignment){
	Subindex subindex(7);
	subindex *= 2;

	EXPECT_EQ(subindex, 14);
}

//TBTKFeature Core.Subindex.operatorMultiplication.1.C++ 2019-09-22
TEST(Subindex, operatorMultiplication){
	Subindex subindex0(7);
	Subindex subindex1(2);

	EXPECT_EQ(subindex0*subindex1, 14);
}

//TBTKFeature Core.Subindex.operatorDivisionAssignment.1.C++ 2019-09-22
TEST(Subindex, operatorDivisionAssignment){
	Subindex subindex(6);
	subindex /= 2;

	EXPECT_EQ(subindex, 3);
}

//TBTKFeature Core.Subindex.operatorDivision.1.C++ 2019-09-22
TEST(Subindex, operatorDivision){
	Subindex subindex0(6);
	Subindex subindex1(2);

	EXPECT_EQ(subindex0/subindex1, 3);
}

//TBTKFeature Core.Subindex.operatorPreIncrement.1.C++ 2019-09-22
TEST(Subindex, operatorIncrement0){
	Subindex subindex(7);

	EXPECT_EQ(++subindex, 8);
}

//TBTKFeature Core.Subindex.operatorPostIncrement.1.C++ 2019-09-22
TEST(Subindex, operatorIncrement1){
	Subindex subindex(7);

	EXPECT_EQ(subindex++, 7);
	EXPECT_EQ(subindex, 8);
}

//TBTKFeature Core.Subindex.operatorPreDecrement.1.C++ 2019-09-22
TEST(Subindex, operatorDecrement0){
	Subindex subindex(7);

	EXPECT_EQ(--subindex, 6);
}

//TBTKFeature Core.Subindex.operatorPostDecrement.1.C++ 2019-09-22
TEST(Subindex, operatorDecrement1){
	Subindex subindex(7);

	EXPECT_EQ(subindex--, 7);
	EXPECT_EQ(subindex, 6);
}

TEST(Subindex, operatorComparsion){
	Subindex subindex0(7);
	Subindex subindex1(7);
	Subindex subindex2(8);

	//TBTKFeature Core.Subindex.operatorComparison.1.C++ 2019-09-22
	EXPECT_EQ(subindex0, subindex1);
	//TBTKFeature Core.Subindex.operatorComparison.2.C++ 2019-09-22
	EXPECT_FALSE(subindex0 == subindex2);
}

TEST(Subindex, operatorNotEqual){
	Subindex subindex0(7);
	Subindex subindex1(7);
	Subindex subindex2(8);

	//TBTKFeature Core.Subindex.operatorNotEqual.1.C++ 2019-09-22
	EXPECT_FALSE(subindex0 != subindex1);
	//TBTKFeature Core.Subindex.operatorNotEqual.2.C++ 2019-09-22
	EXPECT_TRUE(subindex0 != subindex2);
}

TEST(Subindex, operatorLessThan0){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLessThan.1.C++ 2019-09-22
	EXPECT_TRUE(subindex < 8);
	//TBTKFeature Core.Subindex.operatorLessThan.2.C++ 2019-09-22
	EXPECT_FALSE(subindex < 7);
	//TBTKFeature Core.Subindex.operatorLessThan.3.C++ 2019-09-22
	EXPECT_FALSE(subindex < 6);
}

TEST(Subindex, operatorLessThan1){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLessThan.1.C++ 2019-09-22
	EXPECT_TRUE(6 < subindex);
	//TBTKFeature Core.Subindex.operatorLessThan.2.C++ 2019-09-22
	EXPECT_FALSE(7 < subindex);
	//TBTKFeature Core.Subindex.operatorLessThan.3.C++ 2019-09-22
	EXPECT_FALSE(8 < subindex);
}

TEST(Subindex, operatorLargerThan0){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLargerThan.1.C++ 2019-09-22
	EXPECT_TRUE(subindex > 6);
	//TBTKFeature Core.Subindex.operatorLargerThan.2.C++ 2019-09-22
	EXPECT_FALSE(subindex > 7);
	//TBTKFeature Core.Subindex.operatorLargerThan.3.C++ 2019-09-22
	EXPECT_FALSE(subindex > 8);
}

TEST(Subindex, operatorLargerThan1){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLargerThan.1.C++ 2019-09-22
	EXPECT_TRUE(8 > subindex);
	//TBTKFeature Core.Subindex.operatorLargerThan.2.C++ 2019-09-22
	EXPECT_FALSE(7 > subindex);
	//TBTKFeature Core.Subindex.operatorLargerThan.3.C++ 2019-09-22
	EXPECT_FALSE(6 > subindex);
}

TEST(Subindex, operatorLessOrEqualTo0){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLessOrEqualTo.1.C++ 2019-09-22
	EXPECT_TRUE(subindex <= 8);
	//TBTKFeature Core.Subindex.operatorLessOrEqualTo.2.C++ 2019-09-22
	EXPECT_TRUE(subindex <= 7);
	//TBTKFeature Core.Subindex.operatorLessOrEqualTo.3.C++ 2019-09-22
	EXPECT_FALSE(subindex <= 6);
}

TEST(Subindex, operatorLessOrEqualTo1){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLessOrEqualTo.1.C++ 2019-09-22
	EXPECT_TRUE(6 <= subindex);
	//TBTKFeature Core.Subindex.operatorLessOrEqualTo.2.C++ 2019-09-22
	EXPECT_TRUE(7 <= subindex);
	//TBTKFeature Core.Subindex.operatorLessOrEqualTo.3.C++ 2019-09-22
	EXPECT_FALSE(8 <= subindex);
}

TEST(Subindex, operatorLargerOrEqualTo0){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLargerOrEqualTo.1.C++ 2019-09-22
	EXPECT_TRUE(subindex >= 6);
	//TBTKFeature Core.Subindex.operatorLargerOrEqualTo.2.C++ 2019-09-22
	EXPECT_TRUE(subindex >= 7);
	//TBTKFeature Core.Subindex.operatorLargerOrEqualTo.3.C++ 2019-09-22
	EXPECT_FALSE(subindex >= 8);
}

TEST(Subindex, operatorLargerOrEqualTo1){
	Subindex subindex(7);

	//TBTKFeature Core.Subindex.operatorLargerOrEqualTo.1.C++ 2019-09-22
	EXPECT_TRUE(8 >= subindex);
	//TBTKFeature Core.Subindex.operatorLargerOrEqualTo.2.C++ 2019-09-22
	EXPECT_TRUE(7 >= subindex);
	//TBTKFeature Core.Subindex.operatorLargerOrEqualTo.3.C++ 2019-09-22
	EXPECT_FALSE(6 >= subindex);
}

TEST(Subindex, operatorFunction){
	Subindex subindex0(IDX_ALL_);
	Subindex subindex1(IDX_ALL);
	Subindex subindex2(0);
	Subindex subindex3(1);

	//TBTKFeature Core.Subindex.operatorFunction.1.C++ 2019-09-22
	subindex0(1);
	EXPECT_EQ(subindex0, IDX_ALL_(1));

	//TBTKFeature Core.Subindex.operatorFunction.2.C++ 2019-09-22
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			subindex1(1);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//TBTKFeature Core.Subindex.operatorFunction.2.C++ 2019-09-22
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			subindex2(1);
		},
		::testing::ExitedWithCode(1),
		""
	);
	//TBTKFeature Core.Subindex.operatorFunction.2.C++ 2019-09-22
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			subindex3(1);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Subindex , operatorOstream){
	Subindex subindex(7);
	std::stringstream ss;
	ss << subindex;

	EXPECT_TRUE(ss.str().compare("7") == 0);
}

TEST(Subindex, operatorIstream){
	Subindex subindex;
	std::stringstream ss("7");
	ss >> subindex;

	EXPECT_EQ(subindex, 7);
}

#endif

};
