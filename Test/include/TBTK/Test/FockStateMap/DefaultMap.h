#include "TBTK/BitRegister.h"
#include "TBTK/FockStateMap/DefaultMap.h"

#include "gtest/gtest.h"

#include <cmath>

namespace TBTK{
namespace FockStateMap{

//TBTKFeature FockStateMap.DefaultMap.getBasisSize.1 2019-11-04
TEST(DefaultMapMap, getBasisSize1){
	DefaultMap<BitRegister> defaultMap(10);
	EXPECT_EQ(defaultMap.getBasisSize(), pow(2, 10));
}

//TBTKFeature FockStateMap.DefaultMap.getBasisIndex.1 2019-11-04
TEST(DefaultMapMap, getBasisIndex1){
	DefaultMap<BitRegister> defaultMap(10);
	FockState<BitRegister> fockState(10);
	BitRegister &bitRegister = fockState.getBitRegister();
	bitRegister.setBit(3, 1);
	bitRegister.setBit(7, 1);
	EXPECT_EQ(defaultMap.getBasisIndex(fockState), pow(2, 3) + pow(2, 7));
}

//TBTKFeature FockStateMap.DefaultMap.getFockState.1 2019-11-04
TEST(DefaultMapMap, geFockState1){
	DefaultMap<BitRegister> defaultMap(10);
	const FockState<BitRegister> fockState
		= defaultMap.getFockState(pow(2, 3) + pow(2, 7));

	const BitRegister &bitRegister = fockState.getBitRegister();
	for(unsigned int n = 0; n < 10; n++){
		switch(n){
		case 3:
		case 7:
			EXPECT_EQ(bitRegister.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(bitRegister.getBit(n), 0);
			break;
		}
	}
}

};
};
