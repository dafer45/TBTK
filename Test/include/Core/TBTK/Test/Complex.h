#include "TBTK/Complex.h"

#include "gtest/gtest.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

TEST(Complex, Constructor0){
	//Not testable on its own.
}

TEST(Complex, Constructor1){
	Complex complex(std::complex<double>(7, 3));

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, serializeToJSON){
	Complex complex0(std::complex<double>(7, 3));
	Complex complex1(
		complex0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	std::complex<double> c = complex1;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, operatorStdComplexDouble){
	Complex complex(std::complex<double>(7, 3));

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, operatorAssignment){
	Complex complex;
	complex = std::complex<double>(7, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, operatorAdditionAssignment){
	Complex complex(7);
	complex += std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, operatorSubtractionAssignment){
	Complex complex(7);
	complex -= std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), -3);
}

TEST(Complex, operatorMultiplicationAssignment){
	Complex complex(7);
	complex *= std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 0);
	EXPECT_EQ(std::imag(c), 21);
}

TEST(Complex, operatorDivisionAssignment){
	Complex complex(6);
	complex /= std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 0);
	EXPECT_EQ(std::imag(c), -2);
}

#endif

};
