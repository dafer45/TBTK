#include "TBTK/BitRegister.h"
#include "TBTK/FockState.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

//The BitRegister has size 8*sizeof(unsigned int) independently of the size
//passed to the constructor. Also, one bit in the FockState is used to
//indicated whether the state has been killed, i.e., whether it is 0 (which is
//different from |0>). For this reason we use 8*sizeof(unsigned int)-1 as the
//exponential dimension.
unsigned int EXPONENTIAL_DIMENSION = 8*sizeof(unsigned int)-1;

//TBTKFeature ManyParticle.FockState.construction.1 2019-11-03
TEST(FockState, construction1){
	FockState<BitRegister> fockState(EXPONENTIAL_DIMENSION);
	EXPECT_FLOAT_EQ(fockState.getPrefactor(), 1);
}

//TBTKFeature ManyParticle.FockState.isNull.1 2019-11-03
TEST(FockState, isNull1){
	FockState<BitRegister> fockState(EXPONENTIAL_DIMENSION);
	EXPECT_FALSE(fockState.isNull());

	BitRegister &bitRegister = fockState.getBitRegister();
	bitRegister.setBit(bitRegister.getNumBits()-1, 1);
	EXPECT_TRUE(fockState.isNull());
}

//TBTKFeature ManyParticle.FockState.getRegister.1 2019-11-03
TEST(FockState, getRegister1){
	FockState<BitRegister> fockState(EXPONENTIAL_DIMENSION);

	BitRegister &bitRegister0 = fockState.getBitRegister();
	BitRegister &bitRegister1 = fockState.getBitRegister();

	bitRegister0.setBit(7, 1);
	EXPECT_EQ(bitRegister1.getBit(7), 1);
	bitRegister0.setBit(7, 0);
	EXPECT_EQ(bitRegister1.getBit(7), 0);
}

//TBTKFeature ManyParticle.FockState.getRegister.2 2019-11-03
TEST(FockState, getRegister2){
	FockState<BitRegister> fockState(EXPONENTIAL_DIMENSION);

	BitRegister &bitRegister0 = fockState.getBitRegister();
	const BitRegister &bitRegister1 = fockState.getBitRegister();

	bitRegister0.setBit(7, 1);
	EXPECT_EQ(bitRegister1.getBit(7), 1);
	bitRegister0.setBit(7, 0);
	EXPECT_EQ(bitRegister1.getBit(7), 0);
}

//TBTKFeature ManyParticle.FockState.getPrefactor.1 2019-11-03
TEST(FockState, getPrefactor1){
	FockState<BitRegister> fockState(EXPONENTIAL_DIMENSION);
	EXPECT_EQ(fockState.getPrefactor(), 1);
}

};
