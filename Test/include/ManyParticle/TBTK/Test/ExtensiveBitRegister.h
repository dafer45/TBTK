#include "TBTK/ExtensiveBitRegister.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

//TBTKFeature ManyParticle.ExtensiveBitRegister.construction.1 2019-11-03
TEST(ExtensiveBitRegister, construction1){
	ExtensiveBitRegister extensiveBitRegister;
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.construction.2 2019-11-03
TEST(ExtensiveBitRegister, construction2){
	ExtensiveBitRegister extensiveBitRegister(147);
	EXPECT_EQ(
		extensiveBitRegister.getNumBits(),
		8*sizeof(unsigned int)*((147 - 1)/(8*sizeof(unsigned int)) + 1)
	);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.copy.1 2019-11-03
TEST(ExtensiveBitRegister, copy1){
	ExtensiveBitRegister extensiveBitRegister(147);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(13, 1);
	extensiveBitRegister.setBit(47, 1);
	extensiveBitRegister.setBit(102, 1);
	ExtensiveBitRegister copy = extensiveBitRegister;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 1:
		case 13:
		case 47:
		case 102:
			EXPECT_EQ(copy.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(copy.getBit(n), 0);
			break;
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorBitwiseOr.1 2019-11-03
TEST(ExtensiveBitRegister, operatorBitwiseOr1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	ExtensiveBitRegister result
		= extensiveBitRegister0 | extensiveBitRegister1;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 64:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		case 1:
		case 65:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		case 2:
		case 66:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorBitwiseAnd.1 2019-11-03
TEST(ExtensiveBitRegister, operatorBitwiseAnd1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	ExtensiveBitRegister result
		= extensiveBitRegister0 & extensiveBitRegister1;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 64:
			EXPECT_EQ(result.getBit(n), 0);
			break;
		case 1:
		case 65:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		case 2:
		case 66:
			EXPECT_EQ(result.getBit(n), 0);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorBitwiseXor.1 2019-11-03
TEST(ExtensiveBitRegister, operatorBitwiseXor1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	ExtensiveBitRegister result
		= extensiveBitRegister0 ^ extensiveBitRegister1;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 64:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		case 1:
		case 65:
			EXPECT_EQ(result.getBit(n), 0);
			break;
		case 2:
		case 66:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorAddition.1 2019-11-03
TEST(ExtensiveBitRegister, operatorAddition1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	ExtensiveBitRegister result
		= extensiveBitRegister0 + extensiveBitRegister1;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 3:
		case 64:
		case 67:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorSubtraction.1 2019-11-03
TEST(ExtensiveBitRegister, operatorSubtraction1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	ExtensiveBitRegister result
		= extensiveBitRegister1 - extensiveBitRegister0;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorLessThan.1 2019-11-03
TEST(ExtensiveBitRegister, operatorLessThan1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(104, 1);
	extensiveBitRegister1.setBit(5, 1);

	EXPECT_FALSE(extensiveBitRegister0 < extensiveBitRegister1);
	EXPECT_TRUE(extensiveBitRegister1 < extensiveBitRegister0);
	EXPECT_FALSE(extensiveBitRegister0 < extensiveBitRegister0);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorGreaterThan.1 2019-11-03
TEST(ExtensiveBitRegister, operatorGreaterThan1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(104, 1);
	extensiveBitRegister1.setBit(5, 1);

	EXPECT_TRUE(extensiveBitRegister0 > extensiveBitRegister1);
	EXPECT_FALSE(extensiveBitRegister1 > extensiveBitRegister0);
	EXPECT_FALSE(extensiveBitRegister0 > extensiveBitRegister0);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorEqual.1 2019-11-03
TEST(ExtensiveBitRegister, operatorEqual1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(104, 1);
	extensiveBitRegister1.setBit(5, 1);

	EXPECT_FALSE(extensiveBitRegister0 == extensiveBitRegister1);
	EXPECT_FALSE(extensiveBitRegister1 == extensiveBitRegister0);
	EXPECT_TRUE(extensiveBitRegister0 == extensiveBitRegister0);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorAdditionAssignment.1 2019-11-03
TEST(ExtensiveBitRegister, operatorAdditionAssignment1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	extensiveBitRegister0 += extensiveBitRegister1;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 3:
		case 64:
		case 67:
			EXPECT_EQ(extensiveBitRegister0.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister0.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorSubtractionAssignment.1 2019-11-03
TEST(ExtensiveBitRegister, operatorSubtractiobAssignment1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister0.setBit(0, 1);
	extensiveBitRegister0.setBit(1, 1);
	extensiveBitRegister0.setBit(64, 1);
	extensiveBitRegister0.setBit(65, 1);
	extensiveBitRegister1.setBit(1, 1);
	extensiveBitRegister1.setBit(2, 1);
	extensiveBitRegister1.setBit(65, 1);
	extensiveBitRegister1.setBit(66, 1);

	extensiveBitRegister1 -= extensiveBitRegister0;
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(extensiveBitRegister1.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister1.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorPreIncrement.1 2019-11-03
TEST(ExtensiveBitRegister, operatorPreIncrement1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);

	ExtensiveBitRegister returnValue = ++extensiveBitRegister;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 2:
		case 64:
		case 65:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 1);
			EXPECT_EQ(returnValue.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
			EXPECT_EQ(returnValue.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorPostIncrement.1 2019-11-03
TEST(ExtensiveBitRegister, operatorPostIncrement1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);

	ExtensiveBitRegister returnValue = extensiveBitRegister++;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 2:
		case 64:
		case 65:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
		}
	}
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 0:
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(returnValue.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(returnValue.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorPreDecrement.1 2019-11-03
TEST(ExtensiveBitRegister, operatorPreDecrement1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);

	ExtensiveBitRegister returnValue = --extensiveBitRegister;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 1);
			EXPECT_EQ(returnValue.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
			EXPECT_EQ(returnValue.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorPostDecrement.1 2019-11-03
TEST(ExtensiveBitRegister, operatorPostDecrement1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);

	ExtensiveBitRegister returnValue = extensiveBitRegister--;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
		}
	}
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 0:
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(returnValue.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(returnValue.getBit(n), 0);
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorAssignment.1 2019-11-03
TEST(ExtensiveBitRegister, operatorAssignment1){
	ExtensiveBitRegister extensiveBitRegister(147);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(13, 1);
	extensiveBitRegister.setBit(47, 1);
	extensiveBitRegister.setBit(102, 1);
	ExtensiveBitRegister copy(147);
	copy = extensiveBitRegister;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 1:
		case 13:
		case 47:
		case 102:
			EXPECT_EQ(copy.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(copy.getBit(n), 0);
			break;
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorAssignment.2 2019-11-03
TEST(ExtensiveBitRegister, operatorAssignment2){
	ExtensiveBitRegister extensiveBitRegister(147);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(13, 1);
	extensiveBitRegister.setBit(47, 1);
	extensiveBitRegister.setBit(102, 1);
	ExtensiveBitRegister copy;
	copy = extensiveBitRegister;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 1:
		case 13:
		case 47:
		case 102:
			EXPECT_EQ(copy.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(copy.getBit(n), 0);
			break;
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorAssignment.3 2019-11-03
TEST(ExtensiveBitRegister, operatorAssignment3){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			ExtensiveBitRegister extensiveBitRegister(147);
			ExtensiveBitRegister copy(74);
			copy = extensiveBitRegister;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorAssignmentUnsignedInt.1 2019-11-03
TEST(ExtensiveBitRegister, operatorAssignmentUnsignedInt1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister = 0xFF00FF00;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		if((n >= 8 && n < 16) || (n >= 24 && n < 32))
			EXPECT_EQ(extensiveBitRegister.getBit(n), 1);
		else
			EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorLeftBitShift.1 2019-11-03
TEST(ExtensiveBitRegister, operatorLeftBitShift1){
	ExtensiveBitRegister extensiveBitRegister(147);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);

	ExtensiveBitRegister result = extensiveBitRegister << 2;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 2:
		case 3:
		case 66:
		case 67:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
			break;
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.operatorRightBitShift.1 2019-11-03
TEST(ExtensiveBitRegister, operatorRightBitShift1){
	ExtensiveBitRegister extensiveBitRegister(147);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);

	ExtensiveBitRegister result = extensiveBitRegister >> 2;
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 62:
		case 63:
			EXPECT_EQ(result.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(result.getBit(n), 0);
			break;
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.setGetBit.1 2019-11-03
TEST(ExtensiveBitRegister, setGetBit1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(0, 1);
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(64, 1);
	extensiveBitRegister.setBit(65, 1);
	extensiveBitRegister.setBit(66, 1);
	extensiveBitRegister.setBit(66, 0);
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++){
		switch(n){
		case 0:
		case 1:
		case 64:
		case 65:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 1);
			break;
		default:
			EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
			break;
		}
	}
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.toBool.1 2019-11-03
TEST(ExtensiveBitRegister, toBool1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	ExtensiveBitRegister extensiveBitRegister2(128);
	ExtensiveBitRegister extensiveBitRegister3(128);
	ExtensiveBitRegister extensiveBitRegister4(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	extensiveBitRegister2.clear();
	extensiveBitRegister3.clear();
	extensiveBitRegister4.clear();
	extensiveBitRegister1.setBit(0, 1);
	extensiveBitRegister2.setBit(13, 1);
	extensiveBitRegister3.setBit(67, 1);
	extensiveBitRegister4.setBit(127, 1);
	EXPECT_FALSE(extensiveBitRegister0.toBool());
	EXPECT_TRUE(extensiveBitRegister1.toBool());
	EXPECT_TRUE(extensiveBitRegister2.toBool());
	EXPECT_TRUE(extensiveBitRegister3.toBool());
	EXPECT_TRUE(extensiveBitRegister4.toBool());
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.toUnsignedInt.1 2019-11-03
TEST(ExtensiveBitRegister, toUnsignedInt1){
	//Not tested.
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.clear.1 2019-11-03
TEST(ExtensiveBitRegister, clear1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.setBit(27, 1);
	extensiveBitRegister.clear();
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++)
		EXPECT_EQ(extensiveBitRegister.getBit(n), 0);
}

TEST(ExtensiveBitRegister, print1){
	//Not testable.
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.getNumBits.1 2019-11-03
TEST(ExtensiveBitRegister, getNumBits1){
	ExtensiveBitRegister extensiveBitRegister(128);
	EXPECT_EQ(extensiveBitRegister.getNumBits(), 128);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.getNumOneBits.1 2019-11-03
TEST(ExtensiveBitRegister, getNumOneBits1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setBit(1, 1);
	extensiveBitRegister.setBit(13, 1);
	extensiveBitRegister.setBit(47, 1);
	extensiveBitRegister.setBit(67, 1);
	extensiveBitRegister.setBit(127, 1);
	EXPECT_EQ(extensiveBitRegister.getNumOneBits(), 5);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.getMostSignificantBit.1 2019-11-03
TEST(ExtensiveBitRegister, getMostSignificantBit1){
	ExtensiveBitRegister extensiveBitRegister0(128);
	ExtensiveBitRegister extensiveBitRegister1(128);
	extensiveBitRegister0.clear();
	extensiveBitRegister1.clear();
	for(unsigned int n = 0; n < extensiveBitRegister0.getNumBits(); n++){
		switch(n){
		case 0:
			extensiveBitRegister0.setBit(n, 1);
			extensiveBitRegister1.setBit(n, 0);
			break;
		case 127:
			extensiveBitRegister0.setBit(n, 0);
			extensiveBitRegister1.setBit(n, 1);
			break;
		default:
			extensiveBitRegister0.setBit(n, 1);
			extensiveBitRegister1.setBit(n, 1);
			break;
		}
	}
	EXPECT_EQ(extensiveBitRegister0.getMostSignificantBit(), 0);
	EXPECT_EQ(extensiveBitRegister1.getMostSignificantBit(), 1);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.setMostSignificantBit.1 2019-11-03
TEST(ExtensiveBitRegister, setMostSignificantBit1){
	ExtensiveBitRegister extensiveBitRegister(128);
	extensiveBitRegister.clear();
	extensiveBitRegister.setMostSignificantBit();
	EXPECT_EQ(extensiveBitRegister.getMostSignificantBit(), 1);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.clearMostSignificantBit.1 2019-11-03
TEST(ExtensiveBitRegister, clearMostSignificantBit1){
	ExtensiveBitRegister extensiveBitRegister(128);
	for(unsigned int n = 0; n < extensiveBitRegister.getNumBits(); n++)
		extensiveBitRegister.setBit(n, 1);
	extensiveBitRegister.clearMostSignificantBit();
	EXPECT_EQ(extensiveBitRegister.getMostSignificantBit(), 0);
}

//TBTKFeature ManyParticle.ExtensiveBitRegister.cloneStructure.1 2019-11-02
TEST(ExtensiveBitRegister, cloneStructure1){
	ExtensiveBitRegister extensiveBitRegister(128);
	ExtensiveBitRegister result = extensiveBitRegister.cloneStructure();
	EXPECT_EQ(result.getNumBits(), extensiveBitRegister.getNumBits());
}

};
