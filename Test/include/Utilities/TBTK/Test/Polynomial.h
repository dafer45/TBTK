#include "TBTK/Polynomial.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Polynomial, Constructor){
	//Not testable on its own.
}

TEST(Polynomial, addTerm){
	Polynomial<> polynomial(2);

	//Fail for wrong number of variables.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			polynomial.addTerm(1, {1});
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Succeedd for the correct number of variables.
	polynomial.addTerm(1, {1, 2});
}

TEST(Polynomial, functionOperator){
	Polynomial<> polynomial(2);

	polynomial.addTerm(2, {1, 2});
	polynomial.addTerm(3, {0, 1});
	polynomial.addTerm(4, {3, 1});

	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 10; y++){
			EXPECT_DOUBLE_EQ(
				real(polynomial({(double)x, (double)y})),
				2*x*y*y + 3*y + 4*x*x*x*y
			);
		}
	}
}

};
