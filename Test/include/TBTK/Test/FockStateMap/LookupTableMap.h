#include "TBTK/BitRegister.h"
#include "TBTK/FockStateMap/LookupTableMap.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace FockStateMap{

class LookupTableMapTest : public ::testing::Test{
protected:
	LookupTableMap<BitRegister> lookupTableMap;
	FockState<BitRegister> fockState0;
	FockState<BitRegister> fockState1;
	FockState<BitRegister> fockState2;
	const unsigned int EXPONENTIAL_DIMENSION = 10;

	LookupTableMapTest() :
		lookupTableMap(EXPONENTIAL_DIMENSION),
		fockState0(EXPONENTIAL_DIMENSION),
		fockState1(EXPONENTIAL_DIMENSION),
		fockState2(EXPONENTIAL_DIMENSION)
	{
	}

	void SetUp() override{
		fockState0.getBitRegister().setBit(2, 1);
		fockState1.getBitRegister().setBit(7, 1);
		fockState2.getBitRegister().setBit(5, 1);
		lookupTableMap.addState(fockState0);
		lookupTableMap.addState(fockState1);
		lookupTableMap.addState(fockState2);
	}
};

//TBTKFeature FockStateMap.LookupTable.construction.1 2019-11-04
TEST(LookupTableMap, construction1){
	LookupTableMap<BitRegister> lookupTableMap(10);
	EXPECT_EQ(lookupTableMap.getBasisSize(), 0);
}

//TBTKFeature FockStateMap.LookupTable.getBasisSize.1 2019-11-04
TEST_F(LookupTableMapTest, getBasisSize1){
	EXPECT_EQ(lookupTableMap.getBasisSize(), 3);
}

//TBTKFeature FockStateMap.LookupTable.getBasisIndex.1 2019-11-04
TEST_F(LookupTableMapTest, getBasisIndex){
	EXPECT_EQ(lookupTableMap.getBasisIndex(fockState0), 0);
	EXPECT_EQ(lookupTableMap.getBasisIndex(fockState1), 2);
	EXPECT_EQ(lookupTableMap.getBasisIndex(fockState2), 1);
}

//TBTKFeature FockStateMap.LookupTable.addState.1 2019-11-04
TEST(LookupTableMap, addState1){
	const unsigned int EXPONENTIAL_DIMENSION = 10;
	LookupTableMap<BitRegister> lookupTableMap(EXPONENTIAL_DIMENSION);
	FockState<BitRegister> fockState0(EXPONENTIAL_DIMENSION);
	FockState<BitRegister> fockState1(EXPONENTIAL_DIMENSION);
	FockState<BitRegister> fockState2(EXPONENTIAL_DIMENSION);
	fockState0.getBitRegister().setBit(2, 1);
	fockState1.getBitRegister().setBit(7, 1);
	fockState2.getBitRegister().setBit(5, 1);
	lookupTableMap.addState(fockState0);
	lookupTableMap.addState(fockState1);
	lookupTableMap.addState(fockState2);
	EXPECT_EQ(lookupTableMap.getBasisSize(), 3);
	EXPECT_EQ(lookupTableMap.getBasisIndex(fockState0), 0);
	EXPECT_EQ(lookupTableMap.getBasisIndex(fockState1), 2);
	EXPECT_EQ(lookupTableMap.getBasisIndex(fockState2), 1);
}

};
};
