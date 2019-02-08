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

TEST(Integer, serializeToJSON){
	Subindex subindex0(7);
	Subindex subindex1(
		subindex0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(subindex0, subindex1);
}

TEST(Subindex, operatorAssignment){
	Subindex subindex0(7);
	Subindex subindex1(0);
	Subindex subindex2(0);
	subindex1 = (int)7;
	subindex2 = (unsigned int)7;

	EXPECT_EQ(subindex0, subindex1);
	EXPECT_EQ(subindex0, subindex2);
}

TEST(Subindex, operatorComparsion){
	Subindex subindex0(7);
	Subindex subindex1(7);
	Subindex subindex2(8);

	EXPECT_EQ(subindex0, subindex1);
	EXPECT_FALSE(subindex0 == subindex2);
}

#endif

};
