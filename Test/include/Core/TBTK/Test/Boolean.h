#include "TBTK/Boolean.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Boolean, Constructor0){
	//Not testable on its own.
}

TEST(Boolean, Constructor1){
	Boolean boolean(true);

	EXPECT_EQ(boolean, true);
}

TEST(Boolean, serializeToJSON){
	Boolean boolean0(true);
	Boolean boolean1(
		boolean0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(boolean1, true);
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
