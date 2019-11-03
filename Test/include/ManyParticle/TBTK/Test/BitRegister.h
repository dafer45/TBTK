#include "TBTK/BitRegister.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

//TBTKFeature ManyParticle.BitRegister.construction.1 2019-11-02
TEST(BitRegister, construction1){
	BitRegister bitRegister;
}

//TBTKFeature ManyParticle.BitRegister.copy.1 2019-11-02
TEST(BitRegister, copy1){
	BitRegister bitRegister;
	bitRegister = 0xFF00FF00;
	BitRegister copy = bitRegister;
	EXPECT_EQ(copy.getValues(), 0xFF00FF00);
}

//TBTKFeature ManyParticle.BitRegister.operatorBitwiseOr.1 2019-11-02
TEST(BitRegister, operatorBitwiseOr1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0x0000FF00;
	bitRegister1 = 0x0000F0F0;

	BitRegister result = bitRegister0 | bitRegister1;
	EXPECT_EQ(result.getValues(), 0x0000FFF0);
}

//TBTKFeature ManyParticle.BitRegister.operatorBitwiseAnd.1 2019-11-02
TEST(BitRegister, operatorBitwiseAnd1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0x0000FF00;
	bitRegister1 = 0x0000F0F0;

	BitRegister result = bitRegister0 & bitRegister1;
	EXPECT_EQ(result.getValues(), 0x0000F000);
}

//TBTKFeature ManyParticle.BitRegister.operatorBitwiseXor.1 2019-11-02
TEST(BitRegister, operatorBitwiseXor1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0x0000FF00;
	bitRegister1 = 0x0000F0F0;

	BitRegister result = bitRegister0 ^ bitRegister1;
	EXPECT_EQ(result.getValues(), 0x00000FF0);
}

//TBTKFeature ManyParticle.BitRegister.operatorAddition.1 2019-11-02
TEST(BitRegister, operatorAddition1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0x12345678;
	bitRegister1 = 0xABCD1234;

	BitRegister result = bitRegister0 + bitRegister1;
	EXPECT_EQ(result.getValues(), 0x12345678 + 0xABCD1234);
}

//TBTKFeature ManyParticle.BitRegister.operatorSubtraction.1 2019-11-02
TEST(BitRegister, operatorSubtraction1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0xABCD1234;
	bitRegister1 = 0x12345678;

	BitRegister result = bitRegister0 - bitRegister1;
	EXPECT_EQ(result.getValues(), 0xABCD1234 - 0x12345678);
}

//TBTKFeature ManyParticle.BitRegister.operatorLessThan.1 2019-11-02
TEST(BitRegister, operatorLessThan1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0xABCD1234;
	bitRegister1 = 0x12345678;

	EXPECT_FALSE(bitRegister0 < bitRegister1);
	EXPECT_TRUE(bitRegister1 < bitRegister0);
	EXPECT_FALSE(bitRegister0 < bitRegister0);
}

//TBTKFeature ManyParticle.BitRegister.operatorGreaterThan.1 2019-11-02
TEST(BitRegister, operatorGreaterThan1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0xABCD1234;
	bitRegister1 = 0x12345678;

	EXPECT_TRUE(bitRegister0 > bitRegister1);
	EXPECT_FALSE(bitRegister1 > bitRegister0);
	EXPECT_FALSE(bitRegister0 > bitRegister0);
}

//TBTKFeature ManyParticle.BitRegister.operatorEqual.1 2019-11-02
TEST(BitRegister, operatorEqual1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0xABCD1234;
	bitRegister1 = 0x12345678;

	EXPECT_FALSE(bitRegister0 == bitRegister1);
	EXPECT_FALSE(bitRegister1 == bitRegister0);
	EXPECT_TRUE(bitRegister0 == bitRegister0);
}

//TBTKFeature ManyParticle.BitRegister.operatorAdditionAssignment.1 2019-11-02
TEST(BitRegister, operatorAdditionAssignment1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0x12345678;
	bitRegister1 = 0xABCD1234;

	bitRegister0 += bitRegister1;
	EXPECT_EQ(bitRegister0.getValues(), 0x12345678 + 0xABCD1234);
}

//TBTKFeature ManyParticle.BitRegister.operatorSubtractionAssignment.1 2019-11-02
TEST(BitRegister, operatorSubtractionAssignment1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0xABCD1234;
	bitRegister1 = 0x12345678;

	bitRegister0 -= bitRegister1;
	EXPECT_EQ(bitRegister0.getValues(), 0xABCD1234 - 0x12345678);
}

//TBTKFeature ManyParticle.BitRegister.operatorPreIncrement.1 2019-11-02
TEST(BitRegister, operatorPreIncrement1){
	BitRegister bitRegister;
	bitRegister = 0xABCD1234;

	EXPECT_EQ((++bitRegister).getValues(), 0xABCD1234 + 1);
}

//TBTKFeature ManyParticle.BitRegister.operatorPostIncrement.1 2019-11-02
TEST(BitRegister, operatorPostIncrement1){
	BitRegister bitRegister;
	bitRegister = 0xABCD1234;

	EXPECT_EQ((bitRegister++).getValues(), 0xABCD1234);
	EXPECT_EQ(bitRegister.getValues(), 0xABCD1234 + 1);
}

//TBTKFeature ManyParticle.BitRegister.operatorPreDecrement.1 2019-11-02
TEST(BitRegister, operatorPreDecrement1){
	BitRegister bitRegister;
	bitRegister = 0xABCD1234;

	EXPECT_EQ((--bitRegister).getValues(), 0xABCD1234 - 1);
}

//TBTKFeature ManyParticle.BitRegister.operatorPostDecrement.1 2019-11-02
TEST(BitRegister, operatorPostDecrement1){
	BitRegister bitRegister;
	bitRegister = 0xABCD1234;

	EXPECT_EQ((bitRegister--).getValues(), 0xABCD1234);
	EXPECT_EQ(bitRegister.getValues(), 0xABCD1234 - 1);
}

//TBTKFeature ManyParticle.BitRegister.operatorAssignment.1 2019-11-02
TEST(BitRegister, operatorAssignment1){
	BitRegister bitRegister;
	bitRegister = 0xFF00FF00;
	BitRegister copy;
	copy = bitRegister;
	EXPECT_EQ(copy.getValues(), 0xFF00FF00);
}

//TBTKFeature ManyParticle.BitRegister.operatorAssignmentUnsignedInt.1 2019-11-02
TEST(BitRegister, operatorAssignmentUnsignedInt1){
	BitRegister bitRegister;
	bitRegister = 0xFF00FF00;
	EXPECT_EQ(bitRegister.getValues(), 0xFF00FF00);
}

//TBTKFeature ManyParticle.BitRegister.operatorLeftBitshift.1 2019-11-02
TEST(BitRegister, operatorLeftBitShift1){
	BitRegister bitRegister;
	bitRegister = 0xFF00FF00;
	BitRegister result = bitRegister << 2;
	EXPECT_EQ(result.getValues(), 0xFF00FF00 << 2);
	EXPECT_EQ(bitRegister.getValues(), 0xFF00FF00);
}

//TBTKFeature ManyParticle.BitRegister.operatorRightBitshift.1 2019-11-02
TEST(BitRegister, operatorRightBitShift1){
	BitRegister bitRegister;
	bitRegister = 0xFF00FF00;
	BitRegister result = bitRegister >> 2;
	EXPECT_EQ(result.getValues(), 0xFF00FF00 >> 2);
	EXPECT_EQ(bitRegister.getValues(), 0xFF00FF00);
}

//TBTKFeature ManyParticle.BitRegister.setBit.1 2019-11-02
TEST(BitRegister, setBit1){
	BitRegister bitRegister;
	bitRegister = 0xFFFF0000;
	bitRegister.setBit(4, 0);
	bitRegister.setBit(5, 1);
	bitRegister.setBit(20, 0);
	bitRegister.setBit(21, 1);
	EXPECT_EQ(bitRegister.getValues(), 0xFFEF0020);
}

//TBTKFeature ManyParticle.BitRegister.getBit.1 2019-11-02
TEST(BitRegister, getBit1){
	BitRegister bitRegister;
	bitRegister = 0xFFFF0000;
	EXPECT_EQ(bitRegister.getBit(17), 1);
	EXPECT_EQ(bitRegister.getBit(16), 1);
	EXPECT_EQ(bitRegister.getBit(15), 0);
	EXPECT_EQ(bitRegister.getBit(14), 0);
}

//TBTKFeature ManyParticle.BitRegister.setGetValues.1 2019-11-02
TEST(BitRegister, setGetValue1){
	BitRegister bitRegister;
	bitRegister.setValues(0xFFFF0000);
	EXPECT_EQ(bitRegister.getValues(), 0xFFFF0000);
}

//TBTKFeature ManyParticle.BitRegister.toBool.1 2019-11-02
TEST(BitRegister, toBool1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	BitRegister bitRegister2;
	BitRegister bitRegister3;
	BitRegister bitRegister4;
	bitRegister0 = 0x00000000;
	bitRegister1 = 0x00000001;
	bitRegister2 = 0x00000020;
	bitRegister3 = 0x00050000;
	bitRegister4 = 0x80000000;
	EXPECT_FALSE(bitRegister0.toBool());
	EXPECT_TRUE(bitRegister1.toBool());
	EXPECT_TRUE(bitRegister2.toBool());
	EXPECT_TRUE(bitRegister3.toBool());
	EXPECT_TRUE(bitRegister4.toBool());
}

//TBTKFeature ManyParticle.BitRegister.toUnsignedInt.1 2019-11-02
TEST(BitRegister, toUnsignedInt1){
	BitRegister bitRegister;
	bitRegister.setValues(0xFFFF0000);
	EXPECT_EQ(bitRegister.toUnsignedInt(), 0xFFFF0000);
}

//TBTKFeature ManyParticle.BitRegister.clear.1 2019-11-02
TEST(BitRegister, clear1){
	BitRegister bitRegister;
	bitRegister.setValues(0xFFFF0000);
	bitRegister.clear();
	EXPECT_EQ(bitRegister.getValues(), 0x00000000);
}

TEST(BitRegister, print1){
	//Not testable.
}

//TBTKFeature ManyParticle.BitRegister.getNumBits.1 2019-11-02
TEST(BitRegister, getNumBits1){
	BitRegister bitRegister;
	EXPECT_EQ(bitRegister.getNumBits(), 8*sizeof(unsigned int));
}

//TBTKFeature ManyParticle.BitRegister.getNumOneBits.1 2019-11-02
TEST(BitRegister, getNumOneBits1){
	BitRegister bitRegister;
	bitRegister = 0x12345678;
	EXPECT_EQ(bitRegister.getNumOneBits(), 13);
}

//TBTKFeature ManyParticle.BitRegister.getMostSignificantBit.1 2019-11-02
TEST(BitRegister, getMostSignificantBit1){
	BitRegister bitRegister0;
	BitRegister bitRegister1;
	bitRegister0 = 0x7FFFFFFF;
	bitRegister1 = 0x80000000;
	EXPECT_EQ(bitRegister0.getMostSignificantBit(), 0);
	EXPECT_EQ(bitRegister1.getMostSignificantBit(), 1);
}

//TBTKFeature ManyParticle.BitRegister.setMostSignificantBit.1 2019-11-02
TEST(BitRegister, setMostSignificantBit1){
	BitRegister bitRegister;
	bitRegister = 0x00000000;
	bitRegister.setMostSignificantBit();
	EXPECT_EQ(bitRegister.getMostSignificantBit(), 1);
}

//TBTKFeature ManyParticle.BitRegister.clearMostSignificantBit.1 2019-11-02
TEST(BitRegister, clearMostSignificantBit1){
	BitRegister bitRegister;
	bitRegister = 0xFFFFFFFF;
	bitRegister.clearMostSignificantBit();
	EXPECT_EQ(bitRegister.getMostSignificantBit(), 0);
	bitRegister.clearMostSignificantBit();
	EXPECT_EQ(bitRegister.getMostSignificantBit(), 0);
}

//TBTKFeature ManyParticle.BitRegister.cloneStructure.1 2019-11-02
TEST(BitRegister, cloneStructure1){
	BitRegister bitRegister;
	bitRegister.cloneStructure();
	//Dummy function to provide the same interface as ExtensiveBitRegister.
}

};
