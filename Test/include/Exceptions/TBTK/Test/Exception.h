#include "TBTK/Exception.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

class ExceptionTest : public ::testing::Test{
protected:
	Exception exception;

	void SetUp() override{
		exception = Exception(
			"FunctionName",
			"Where",
			"Message",
			"Hint"
		);
	}
};

//TBTKFeature Exceptions.Exception.what.1 2019-11-19
TEST_F(ExceptionTest, what1){
	EXPECT_EQ(
		std::string(exception.what()).compare(exception.getMessage()),
		0
	);
}

//TBTKFeature Exceptions.Exception.print.1 2019-11-19
TEST_F(ExceptionTest, print1){
	//Not testable.
}

//TBTKFeature Exceptions.Exception.getFunction.1 2019-11-19
TEST_F(ExceptionTest, getFunction1){
	EXPECT_TRUE(exception.getFunction().compare("FunctionName") == 0);
}

//TBTKFeature Exceptions.Exception.getWhere.1 2019-11-19
TEST_F(ExceptionTest, getWhere1){
	EXPECT_TRUE(exception.getWhere().compare("Where") == 0);
}

//TBTKFeature Exceptions.Exception.getMessage.1 2019-11-19
TEST_F(ExceptionTest, getMessage1){
	EXPECT_TRUE(exception.getMessage().compare("Message") == 0);
}

//TBTKFeature Exceptions.Exception.getHint.1 2019-11-19
TEST_F(ExceptionTest, getHint1){
	EXPECT_TRUE(exception.getHint().compare("Hint") == 0);
}

};
