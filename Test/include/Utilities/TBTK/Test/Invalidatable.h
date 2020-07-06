#include "TBTK/Invalidatable.h"
#include "TBTK/Integer.h"
#include "TBTK/CArray.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.Invalidatable.construction.0 2020-07-06
TEST(Invalidatable, construction0){
	Invalidatable<Integer> invalidatable;
}

//TBTKFeature Utilities.Invalidable.construction.1 2020-07-06
TEST(Invalidable, construction1){
	Invalidatable<Integer> invalidatable0(7);
	Invalidatable<Integer> invalidatable1(Integer(7));
}

//TBTKFeature Utilities.Invalidatable.serializeToJSON.0 2020-07-06
TEST(Invalidatable, serializeToJSON0){
	for(unsigned int n = 0; n < 2; n++){
		Invalidatable<Integer> invalidatable;
		invalidatable.setIsValid(n == 0);
		invalidatable = 7;

		Invalidatable<Integer> reserialized(
			invalidatable.serialize(Serializable::Mode::JSON),
			Serializable::Mode::JSON
		);
		EXPECT_EQ(reserialized.getIsValid(), n == 0);
		EXPECT_EQ(reserialized, 7);
	}
}

//TBTKFeature Utilities.Invalidatable.setGetIsValid.0 2020-07-06
TEST(Invalidatable, setGetIsValid){
	Invalidatable<Integer> invalidatable;
	invalidatable.setIsValid(false);
	EXPECT_FALSE(invalidatable.getIsValid());
	invalidatable.setIsValid(true);
	EXPECT_TRUE(invalidatable.getIsValid());

	Invalidatable<CArray<CArray<double>>> array;
	array = CArray<CArray<double>>(10);
	for(unsigned int n = 0; n < array.getSize(); n++)
		array[n] = CArray<double>(20);
	for(unsigned int n = 0; n < 10; n++){
		for(unsigned int c = 0; c < 10; c++){
			array[n][c] = n*c;
		}
	}
	for(unsigned int n = 0; n < 10; n++)
		for(unsigned int c = 0; c < 10; c++)
			EXPECT_EQ(array[n][c], n*c);
}

};
