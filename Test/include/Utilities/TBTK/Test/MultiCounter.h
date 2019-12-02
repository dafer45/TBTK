#include "TBTK/MultiCounter.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.Array.construction.1 2019-11-02
TEST(MultiCounter, construction1){
	MultiCounter<unsigned int> multiCounter(
		{0, 1, 2},
		{5, 7, 8},
		{1, 2, 3}
	);
	EXPECT_EQ(multiCounter[0], 0);
	EXPECT_EQ(multiCounter[1], 1);
	EXPECT_EQ(multiCounter[2], 2);
}

//TBTKFeature Utilities.Array.operatorIncrement.1 2019-11-02
TEST(MultiCounter, operatorIncrement1){
	MultiCounter<unsigned int> multiCounter(
		{0, 1, 2},
		{5, 7, 8},
		{1, 2, 3}
	);
	for(unsigned int x = 0; x < 5; x++){
		for(unsigned int y = 1; y < 7; y += 2){
			for(unsigned int z = 2; z < 8; z += 3){
				EXPECT_EQ(multiCounter[0], x);
				EXPECT_EQ(multiCounter[1], y);
				EXPECT_EQ(multiCounter[2], z);
				++multiCounter;
			}
		}
	}
}

//TBTKFeature Utilities.Array.operatorStdVector.1 2019-12-02
TEST(MultiCounter, operatorStdVector1){
	MultiCounter<unsigned int> multiCounter(
		{0, 1, 2},
		{5, 7, 8},
		{1, 2, 3}
	);
	for(unsigned int x = 0; x < 5; x++){
		for(unsigned int y = 1; y < 7; y += 2){
			for(unsigned int z = 2; z < 8; z += 3){
				std::vector<unsigned int> v = multiCounter;
				EXPECT_EQ(v[0], x);
				EXPECT_EQ(v[1], y);
				EXPECT_EQ(v[2], z);
				++multiCounter;
			}
		}
	}
}

//TBTKFeature Utilities.Array.reset.1 2019-11-02
TEST(MultiCounter, reset1){
	MultiCounter<unsigned int> multiCounter(
		{0, 1, 2},
		{5, 7, 8},
		{1, 2, 3}
	);
	++multiCounter;
	multiCounter.reset();
	EXPECT_EQ(multiCounter[0], 0);
	EXPECT_EQ(multiCounter[1], 1);
	EXPECT_EQ(multiCounter[2], 2);
}

//TBTKFeature Utilities.Array.deon.1 2019-11-02
TEST(MultiCounter, done1){
	MultiCounter<unsigned int> multiCounter(
		{0, 1, 2},
		{5, 7, 8},
		{1, 2, 3}
	);
	for(unsigned int x = 0; x < 5; x++){
		for(unsigned int y = 1; y < 7; y += 2){
			for(unsigned int y = 2; y < 8; y += 3){
				EXPECT_FALSE(multiCounter.done());
				++multiCounter;
			}
		}
	}
	EXPECT_TRUE(multiCounter.done());
}

};
