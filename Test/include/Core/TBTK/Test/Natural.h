#include "TBTK/Natural.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Natural, Constructor0){
	//Not testable on its own.
}

TEST(Natural, Constructor1){
	Natural natural(7);

	EXPECT_EQ(natural, 7);
}

TEST(Natural, serializeToJSON){
	Natural natural0(7);
	Natural natural1(
		natural0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(natural1, 7);
}

TEST(Natural, operatorUnsignedInt){
	Natural natural(7);

	EXPECT_EQ(natural, 7);
}

TEST(Natural, operatorAssignment){
	Natural natural;
	natural = 7;

	EXPECT_EQ(natural, 7);
}

TEST(Natural, operatorAdditionAssignment){
	Natural natural(7);
	natural += 8;

	EXPECT_EQ(natural, 15);
}

TEST(Natural, operatorSubtractionAssignment){
	Natural natural(7);
	natural -= 8;

	EXPECT_EQ(natural, -1);
}

TEST(Natural, operatorMultiplicationAssignment){
	Natural natural(7);
	natural *= 8;

	EXPECT_EQ(natural, 56);
}

TEST(Natural, operatorDivisionAssignment){
	Natural natural(70);
	natural /= 10;

	EXPECT_EQ(natural, 7);
}

};
