#include "TBTK/ElementNotFoundException.h"

#include "gtest/gtest.h"

#include <sstream>

namespace TBTK{

class ElementNotFoundExceptionTest : public ::testing::Test{
protected:
	ElementNotFoundException exception;

	void SetUp() override{
		exception = ElementNotFoundException(
			"FunctionName",
			"Where",
			"Message",
			"Hint"
		);
	}
};

//TBTKFeature Exceptions.ElementNotFoundException.getFunction.1 2019-11-19
TEST_F(ElementNotFoundExceptionTest, getFunction1){
	EXPECT_TRUE(exception.getFunction().compare("FunctionName") == 0);
}

//TBTKFeature Exceptions.ElementNotFoundException.getWhere.1 2019-11-19
TEST_F(ElementNotFoundExceptionTest, getWhere1){
	EXPECT_TRUE(exception.getWhere().compare("Where") == 0);
}

//TBTKFeature Exceptions.ElementNotFoundException.getMessage.1 2019-11-19
TEST_F(ElementNotFoundExceptionTest, getMessage1){
	EXPECT_TRUE(exception.getMessage().compare("Message") == 0);
}

//TBTKFeature Exceptions.ElementNotFoundException.getHint.1 2019-11-19
TEST_F(ElementNotFoundExceptionTest, getHint1){
	EXPECT_TRUE(exception.getHint().compare("Hint") == 0);
}

};
