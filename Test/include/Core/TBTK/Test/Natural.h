#include "TBTK/Natural.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Natural, Constructor0){
	//Not testable on its own.
}

TEST(Natural, Constructor1){
	Natural natural(7);

	EXPECT_EQ(natural, 7);
}

TEST(Natural, serializeToJSON0){
	Natural natural0(7);
	Natural natural1(
		natural0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(natural1, 7);
}

TEST(Natural, serializeToJSON1){
	Natural natural(7);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			natural.serialize(static_cast<Serializable::Mode>(-1));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Natural, serializeToJSON2){
	Natural natural(7);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Natural(
				natural.serialize(Serializable::Mode::JSON),
				static_cast<Serializable::Mode>(-1)
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
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

TEST(Natural, operatorIncrement0){
	Natural natural(7);

	EXPECT_EQ(++natural, 8);
}

TEST(Natural, operatorIncrement1){
	Natural natural(7);

	EXPECT_EQ(natural++, 7);
	EXPECT_EQ(natural, 8);
}

TEST(Natural, operatorDecrement0){
	Natural natural(7);

	EXPECT_EQ(--natural, 6);
}

TEST(Natural, operatorDecrement1){
	Natural natural(7);

	EXPECT_EQ(natural--, 7);
	EXPECT_EQ(natural, 6);
}

#endif

};
