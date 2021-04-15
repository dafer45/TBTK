#include "TBTK/Boolean.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Boolean, Constructor0){
	//Verify that this compiles.
	Boolean boolean;
}

TEST(Boolean, Constructor1){
	Boolean boolean(true);

	EXPECT_EQ(boolean, true);
}

TEST(Boolean, serializeToJSON0){
	Boolean boolean0(true);
	Boolean boolean1(
		boolean0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(boolean1, true);
}

TEST(Boolean, serializeToJSON1){
	Boolean boolean(true);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			boolean.serialize(static_cast<Serializable::Mode>(-1));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Boolean, serializeToJSON2){
	Boolean boolean(true);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Boolean(
				boolean.serialize(Serializable::Mode::JSON),
				static_cast<Serializable::Mode>(-1)
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Boolean, operatorBool){
	Boolean boolean(true);

	EXPECT_EQ(boolean, true);
}

TEST(Boolean, operatorAssignment){
	Boolean boolean;
	boolean = true;

	EXPECT_EQ(boolean, true);
}

#endif

};
