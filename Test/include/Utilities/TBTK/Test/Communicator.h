#include "TBTK/Communicator.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.Communicator.construction.1 2019-11-01
TEST(Communicator, constructor0){
	Communicator communicatorTrue(true);
	Communicator communicatorFalse(false);

	EXPECT_TRUE(communicatorTrue.getVerbose());
	EXPECT_FALSE(communicatorFalse.getVerbose());
}

//TBTKFeature Utilities.Communicator.setGetVerbose.1 2019-11-01
TEST(Communicator, setGetVerbose1){
	Communicator communicator(true);

	communicator.setVerbose(false);
	EXPECT_FALSE(communicator.getVerbose());

	communicator.setVerbose(false);
	EXPECT_FALSE(communicator.getVerbose());

	communicator.setVerbose(true);
	EXPECT_TRUE(communicator.getVerbose());
}

//TBTKFeature Utilities.Communicator.setGetVerbose.1 2019-11-01
TEST(Communicator, setGetGlobalVerbose1){
	Communicator::setGlobalVerbose(false);
	EXPECT_FALSE(Communicator::getGlobalVerbose());

	Communicator::setGlobalVerbose(true);
	EXPECT_TRUE(Communicator::getGlobalVerbose());
}

};
