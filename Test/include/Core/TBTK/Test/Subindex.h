#include "TBTK/Subindex.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Subindex, Constructor){
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

TEST(Subindex, operatorLessThan){
	Subindex subindex(7);

	EXPECT_TRUE(7 < 8);
	EXPECT_FALSE(7 < 7);
	EXPECT_FALSE(7 < 6);
}

TEST(Subindex, operatorLargerThan){
	Subindex subindex(7);

	EXPECT_TRUE(7 > 6);
	EXPECT_FALSE(7 > 7);
	EXPECT_FALSE(7 > 8);
}

TEST(Subindex , operatorOstream){
	//Not testable.
}

#endif

};
