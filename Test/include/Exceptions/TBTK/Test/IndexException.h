#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

class IndexExceptionTest : public ::testing::Test{
protected:
	IndexException exception;

	void SetUp() override{
		exception = IndexException(
			"FunctionName",
			"Where",
			"Message",
			"Hint"
		);
	}
};

//TBTKFeature Exceptions.IndexException.getFunction.1 2019-11-19
TEST_F(IndexExceptionTest, getFunction1){
	EXPECT_TRUE(exception.getFunction().compare("FunctionName") == 0);
}

//TBTKFeature Exceptions.IndexException.getWhere.1 2019-11-19
TEST_F(IndexExceptionTest, getWhere1){
	EXPECT_TRUE(exception.getWhere().compare("Where") == 0);
}

//TBTKFeature Exceptions.IndexException.getMessage.1 2019-11-19
TEST_F(IndexExceptionTest, getMessage1){
	EXPECT_TRUE(exception.getMessage().compare("Message") == 0);
}

//TBTKFeature Exceptions.IndexException.getHint.1 2019-11-19
TEST_F(IndexExceptionTest, getHint1){
	EXPECT_TRUE(exception.getHint().compare("Hint") == 0);
}

};
