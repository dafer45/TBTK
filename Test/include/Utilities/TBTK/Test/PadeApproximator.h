#include "TBTK/PadeApproximator.h"

#include "TBTK/Array.h"

#include "gtest/gtest.h"

namespace TBTK{

const double EPSILON_1000000 = 1000000*std::numeric_limits<double>::epsilon();

TEST(PadeApproximator, setNumeratorDegree){
	//Not testable on its own.
}

TEST(PadeApproximator, setDenominatorDegree){
	//Not testable on its own.
}

//Helper function for testing PadeApproximator::approximate().
std::complex<double> f(std::complex<double> z){
	return (1. + 3.*z - z*z)/(std::complex<double>(1, 0.1) - z - 4.*z*z + z*z*z);
}

TEST(PadeApproximator, approximate){
	std::vector<std::complex<double>> values;
	std::vector<std::complex<double>> arguments;
	for(unsigned int n = 0; n < 100; n++){
		std::complex<double> z = n/10.;
		values.push_back(f(z));
		arguments.push_back(z);
	}

	PadeApproximator padeApproximator;
	padeApproximator.setNumeratorDegree(3);
	padeApproximator.setDenominatorDegree(4);
	std::vector<Polynomial<>> polynomials
		= padeApproximator.approximate(values, arguments);

	for(unsigned int n = 0; n < 100; n++){
		std::complex<double> z(0, n/10.);

		EXPECT_NEAR(
			real(polynomials[0]({z})/polynomials[1]({z})),
			real(f(z)),
			EPSILON_1000000
		);
		EXPECT_NEAR(
			imag(polynomials[0]({z})/polynomials[1]({z})),
			imag(f(z)),
			EPSILON_1000000
		);
	}
}

};
