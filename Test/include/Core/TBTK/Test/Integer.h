#include "TBTK/Integer.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Integer, Constructor0){
	//Not testable on its own.
}

TEST(Integer, Constructor1){
	Integer integer(7);

	EXPECT_EQ(integer, 7);
}

TEST(Integer, serializeToJSON){
	Integer integer0(7);
	Integer integer1(
		integer0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(integer1, 7);
}

TEST(Integer, operatorInt){
	Integer integer(7);

	EXPECT_EQ(integer, 7);
}

TEST(Integer, operatorAssignment){
	Integer integer;
	integer = 7;

	EXPECT_EQ(integer, 7);
}

TEST(Integer, operatorAdditionAssignment){
	Integer integer(7);
	integer += 8;

	EXPECT_EQ(integer, 15);
}

TEST(Integer, operatorSubtractionAssignment){
	Integer integer(7);
	integer -= 8;

	EXPECT_EQ(integer, -1);
}

TEST(Integer, operatorMultiplicationAssignment){
	Integer integer(7);
	integer *= 8;

	EXPECT_EQ(integer, 56);
}

TEST(Integer, operatorDivisionAssignment){
	Integer integer(70);
	integer /= 10;

	EXPECT_EQ(integer, 7);
}

};
