#include "TBTK/Polynomial.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Polynomial, Constructor){
	//Not testable on its own.
}

TEST(Polynomial, addTerm0){
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

TEST(Polynomial, addTerm1){
	Polynomial<> polynomial(2);

	//Fail for wrong number of variables.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			polynomial.addTerm(Polynomial<>(1), 1);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Succeedd for the correct number of variables.
	polynomial.addTerm(Polynomial<>(2), 1);
}

TEST(Polynomial, addTerm2){
	Polynomial<> polynomial(2);

	//Fail for wrong number of variables.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			polynomial.addTerm(Polynomial<>(1), Polynomial<>(2));
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			polynomial.addTerm(Polynomial<>(2), Polynomial<>(1));
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Succeedd for the correct number of variables.
	polynomial.addTerm(Polynomial<>(2), Polynomial<>(2));
}

TEST(Polynomial, functionOperator){
	Polynomial<> numerator(2);
	numerator.addTerm(1, {0, 0});
	numerator.addTerm(2, {2, 0});
	numerator.addTerm(3, {0, 1});
	numerator.addTerm(1, {1, 1});

	Polynomial<> denominator0(2);
	denominator0.addTerm(4, {0, 0});
	denominator0.addTerm(-1, {1, 0});
	denominator0.addTerm(1, {0, 2});

	Polynomial<> denominator(2);
	denominator.addTerm(denominator0, 2);
	denominator.addTerm(-3, {1, 1});

	Polynomial<> denominatorInverted(2);
	denominatorInverted.addTerm(denominator, -1);

	Polynomial<> polynomial(2);
	polynomial.addTerm(numerator, denominatorInverted);

	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 10; y++){
			EXPECT_DOUBLE_EQ(
				real(polynomial({(double)x, (double)y})),
				(1. + 2.*x*x + 3.*y + x*y)/(pow((4. - x + y*y), 2) - 3.*x*y)
			);
		}
	}
}

};
