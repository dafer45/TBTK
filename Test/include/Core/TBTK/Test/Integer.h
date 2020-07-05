#include "TBTK/Integer.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Integer, Constructor0){
	//Not testable on its own.
}

TEST(Integer, Constructor1){
	Integer integer(7);

	EXPECT_EQ(integer, 7);
}

TEST(Integer, serializeToJSON0){
	Integer integer0(7);
	Integer integer1(
		integer0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(integer1, 7);
}

TEST(Integer, serializeToJSON1){
	Integer integer(7);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			integer.serialize(static_cast<Serializable::Mode>(-1));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Integer, serializeToJSON2){
	Integer integer0(7);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Integer integer1(
				integer0.serialize(Serializable::Mode::JSON),
				static_cast<Serializable::Mode>(-1)
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
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

TEST(Integer, operatorIncrement0){
	Integer integer(7);

	EXPECT_EQ(++integer, 8);
}

TEST(Integer, operatorIncrement1){
	Integer integer(7);

	EXPECT_EQ(integer++, 7);
	EXPECT_EQ(integer, 8);
}

TEST(Integer, operatorDecrement0){
	Integer integer(7);

	EXPECT_EQ(--integer, 6);
}

TEST(Integer, operatorDecrement1){
	Integer integer(7);

	EXPECT_EQ(integer--, 7);
	EXPECT_EQ(integer, 6);
}

#endif

};
