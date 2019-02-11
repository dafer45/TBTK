#include "TBTK/Subindex.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Subindex, Constructor0){
	//Not testable on its own.
}

TEST(Subindex, Constructor1){
	Subindex subindex0((int)7);
	Subindex subindex1(Integer(7));
	Subindex subindex2((unsigned int)7);

	EXPECT_EQ(subindex0, subindex1);
	EXPECT_EQ(subindex0, subindex2);
}

TEST(Subindex, serializeToJSON){
	Subindex subindex0(7);
	Subindex subindex1(
		subindex0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(subindex0, subindex1);
}

TEST(Subindex, operatorInt){
	Subindex subindex(7);
	int i = (int)subindex;

	EXPECT_EQ(i, 7);
}

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

TEST(Subindex, operatorAdditionAssignment){
	Subindex subindex(7);
	subindex += 1;

	EXPECT_EQ(subindex, 8);
}

TEST(Subindex, operatorAddition){
	Subindex subindex0(7);
	Subindex subindex1(1);

	EXPECT_EQ(subindex0 + subindex1, 8);
}

TEST(Subindex, operatorSubtractionAssignment){
	Subindex subindex(7);
	subindex -= 1;

	EXPECT_EQ(subindex, 6);
}

TEST(Subindex, operatorSubtraction){
	Subindex subindex0(7);
	Subindex subindex1(1);

	EXPECT_EQ(subindex0 - subindex1, 6);
}

TEST(Subindex, operatorMultiplicationAssignment){
	Subindex subindex(7);
	subindex *= 2;

	EXPECT_EQ(subindex, 14);
}

TEST(Subindex, operatorMultiplication){
	Subindex subindex0(7);
	Subindex subindex1(2);

	EXPECT_EQ(subindex0*subindex1, 14);
}

TEST(Subindex, operatorDivisionAssignment){
	Subindex subindex(6);
	subindex /= 2;

	EXPECT_EQ(subindex, 3);
}

TEST(Subindex, operatorDivision){
	Subindex subindex0(6);
	Subindex subindex1(2);

	EXPECT_EQ(subindex0/subindex1, 3);
}

TEST(Subindex, operatorIncrement0){
	Subindex subindex(7);

	EXPECT_EQ(++subindex, 8);
}

TEST(Subindex, operatorIncrement1){
	Subindex subindex(7);

	EXPECT_EQ(subindex++, 7);
	EXPECT_EQ(subindex, 8);
}

TEST(Subindex, operatorDecrement0){
	Subindex subindex(7);

	EXPECT_EQ(--subindex, 6);
}

TEST(Subindex, operatorDecrement1){
	Subindex subindex(7);

	EXPECT_EQ(subindex--, 7);
	EXPECT_EQ(subindex, 6);
}

TEST(Subindex, operatorComparsion){
	Subindex subindex0(7);
	Subindex subindex1(7);
	Subindex subindex2(8);

	EXPECT_EQ(subindex0, subindex1);
	EXPECT_FALSE(subindex0 == subindex2);
}

TEST(Subindex, operatorNotEqual){
	Subindex subindex0(7);
	Subindex subindex1(7);
	Subindex subindex2(8);

	EXPECT_FALSE(subindex0 != subindex1);
	EXPECT_TRUE(subindex0 != subindex2);
}

TEST(Subindex, operatorLessThan0){
	Subindex subindex(7);

	EXPECT_TRUE(subindex < 8);
	EXPECT_FALSE(subindex < 7);
	EXPECT_FALSE(subindex < 6);
}

TEST(Subindex, operatorLessThan1){
	Subindex subindex(7);

	EXPECT_TRUE(6 < subindex);
	EXPECT_FALSE(7 < subindex);
	EXPECT_FALSE(8 < subindex);
}

TEST(Subindex, operatorLargerThan0){
	Subindex subindex(7);

	EXPECT_TRUE(subindex > 6);
	EXPECT_FALSE(subindex > 7);
	EXPECT_FALSE(subindex > 8);
}

TEST(Subindex, operatorLargerThan1){
	Subindex subindex(7);

	EXPECT_TRUE(8 > subindex);
	EXPECT_FALSE(7 > subindex);
	EXPECT_FALSE(6 > subindex);
}

TEST(Subindex, operatorLessOrEqualTo0){
	Subindex subindex(7);

	EXPECT_TRUE(subindex <= 8);
	EXPECT_TRUE(subindex <= 7);
	EXPECT_FALSE(subindex <= 6);
}

TEST(Subindex, operatorLessOrEqualTo1){
	Subindex subindex(7);

	EXPECT_TRUE(6 <= subindex);
	EXPECT_TRUE(7 <= subindex);
	EXPECT_FALSE(8 <= subindex);
}

TEST(Subindex, operatorLargerOrEqualTo0){
	Subindex subindex(7);

	EXPECT_TRUE(subindex >= 6);
	EXPECT_TRUE(subindex >= 7);
	EXPECT_FALSE(subindex >= 8);
}

TEST(Subindex, operatorLargerOrEqualTo1){
	Subindex subindex(7);

	EXPECT_TRUE(8 >= subindex);
	EXPECT_TRUE(7 >= subindex);
	EXPECT_FALSE(6 >= subindex);
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
