#include "TBTK/Real.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Real, Constructor0){
	//Not testable on its own.
}

TEST(Real, Constructor1){
	Real real(7);

	EXPECT_EQ(real, 7);
}

TEST(Real, serializeToJSON0){
	Real real0(7);
	Real real1(
		real0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(real1, 7);
}

TEST(Real, serializeToJSON1){
	Real real(7);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			real.serialize(static_cast<Serializable::Mode>(-1));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Real, serializeToJSON2){
	Real real(7);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Real(
				real.serialize(Serializable::Mode::JSON),
				static_cast<Serializable::Mode>(-1)
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Real, operatorDouble){
	Real real(7);

	EXPECT_EQ(real, 7);
}

TEST(Real, operatorAssignment){
	Real real;
	real = 7;

	EXPECT_EQ(real, 7);
}

TEST(Real, operatorAdditionAssignment){
	Real real(7);
	real += 8;

	EXPECT_EQ(real, 15);
}

TEST(Real, operatorSubtractionAssignment){
	Real real(7);
	real -= 8;

	EXPECT_EQ(real, -1);
}

TEST(Real, operatorMultiplicationAssignment){
	Real real(7);
	real *= 8;

	EXPECT_EQ(real, 56);
}

TEST(Real, operatorDivisionAssignment){
	Real real(70);
	real /= 10;

	EXPECT_EQ(real, 7);
}

#endif

};
