#include "TBTK/Range.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.Range.construction.0 2020-07-03
TEST(Range, construction0){
	//Cerify that this compiles.
	Range range;
}

//TBTKFeature Utilities.Range.construction.1 2019-11-02
TEST(Range, construction1){
	Range range(-10, 10, 100);
	EXPECT_FLOAT_EQ(range[0], -10);
	EXPECT_FLOAT_EQ(range[99], 10);
	EXPECT_EQ(range.getResolution(), 100);
}

//TBTKFeature Utilities.Range.construction.2 2019-11-02
TEST(Range, construction2){
	Range range(-10, 10, 100, true, false);
	EXPECT_FLOAT_EQ(range[0], -10);
	EXPECT_FLOAT_EQ(range[99], -10 + 99*20/(double)100);
	EXPECT_EQ(range.getResolution(), 100);
}

//TBTKFeature Utilities.Range.construction.3 2019-11-02
TEST(Range, construction3){
	Range range(-10, 10, 100, false, true);
	EXPECT_FLOAT_EQ(range[0], 10 - 99*20/(double)100);
	EXPECT_FLOAT_EQ(range[99], 10);
	EXPECT_EQ(range.getResolution(), 100);
}

//TBTKFeature Utilities.Range.construction.4 2019-11-02
TEST(Range, construction4){
	Range range(-10, 10, 100, false, false);
	EXPECT_FLOAT_EQ(range[0], -10 + 20/(double)101);
	EXPECT_FLOAT_EQ(range[99], 10 - 20/(double)101);
	EXPECT_EQ(range.getResolution(), 100);
}

//TBTKFeature Utilities.Range.SerializeToJSON.1 2019-11-02
TEST(Range, SerializeToJSON1){
	Range range(-10, 10, 100);
	Range copy(
		range.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_FLOAT_EQ(copy[0], range[0]);
	EXPECT_FLOAT_EQ(copy[99], range[99]);
	EXPECT_FLOAT_EQ(copy.getResolution(), range.getResolution());
}

//TBTKFeature Utilities.Range.getResolution.1 2019-11-02
TEST(Range, getResolution1){
	Range range(-10, 10, 100, false, false);
	EXPECT_EQ(range.getResolution(), 100);
}

//TBTKFeature Utilities.Range.operatorArraySubscript.1 2019-11-02
TEST(Range, operatorArraySubscript1){
	Range range(-10, 10, 100);
	for(unsigned int n = 0; n < range.getResolution(); n++)
		EXPECT_FLOAT_EQ(range[n], -10 + n*20/(double)99);
}

//TBTKFeature Utilities.Range.operatorComparison.1 2020-07-03
TEST(Range, operatorComparison1){
	EXPECT_TRUE(Range(-10, 10, 1000) == Range(-10, 10, 1000));
	EXPECT_FALSE(Range(-10, 10, 1000) == Range(-9, 10, 1000));
	EXPECT_FALSE(Range(-10, 10, 1000) == Range(-10, 9, 1000));
	EXPECT_FALSE(Range(-10, 10, 1000) == Range(-10, 10, 999));
}

//TBTKFeature Utilities.Range.getLast.1 2020-07-03
TEST(Range, getLast1){
	Range range(-10, 10, 100);
	EXPECT_FLOAT_EQ(range.getLast(), range[99]);
}

};
