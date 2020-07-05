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

TEST(Complex, Constructor2){
	Complex complex(7, 3);

	EXPECT_EQ(real(complex), 7);
	EXPECT_EQ(imag(complex), 3);
}

TEST(Complex, Constructor3){
	Complex complex(7);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 0);
}

TEST(Complex, serializeToJSON0){
	Complex complex0(std::complex<double>(7, 3));
	Complex complex1(
		complex0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	std::complex<double> c = complex1;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, serializeToJSON1){
	Complex complex(std::complex<double>(7, 3));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			complex.serialize(static_cast<Serializable::Mode>(-1));
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(Complex, serializeToJSON2){
	Complex complex(std::complex<double>(7, 3));
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Complex(
				complex.serialize(Serializable::Mode::JSON),
				static_cast<Serializable::Mode>(-1)
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
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

TEST(Complex, operatorAdditionAssignment0){
	Complex complex(7);
	complex += std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, operatorAdditionAssignment1){
	std::complex<double> complex(7, 0);
	complex += Complex(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), 3);
}

TEST(Complex, operatorAddition0){
	Complex complex(0, 3);

	EXPECT_EQ(complex + 7, Complex(7, 3));
}

TEST(Complex, operatorAddition1){
	Complex complex(0, 3);

	EXPECT_EQ(7 + complex, Complex(7, 3));
}

TEST(Complex, operatorSubtractionAssignment){
	Complex complex(7);
	complex -= std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 7);
	EXPECT_EQ(std::imag(c), -3);
}

TEST(Complex, operatorSubtraction0){
	EXPECT_EQ(Complex(7, 0) - Complex(0, 3), Complex(7, -3));
}

TEST(Complex, operatorSubtraction1){
	EXPECT_EQ(Complex(0, 3) - 7, Complex(-7, 3));
}

TEST(Complex, operatorSubtraction2){
	EXPECT_EQ(7 - Complex(0, 3), Complex(7, -3));
}

TEST(Complex, operatorUnaryMinus){
	EXPECT_EQ(-Complex(7, 3), Complex(-7, -3));
}

TEST(Complex, operatorMultiplicationAssignment){
	Complex complex(7);
	complex *= std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 0);
	EXPECT_EQ(std::imag(c), 21);
}

TEST(Complex, operatorMultiplication0){
	EXPECT_EQ(Complex(7, 3)*2, Complex(14, 6));
}

TEST(Complex, operatorMultiplication1){
	EXPECT_EQ(2*Complex(7, 3), Complex(14, 6));
}

TEST(Complex, operatorDivisionAssignment){
	Complex complex(6);
	complex /= std::complex<double>(0, 3);

	std::complex<double> c = complex;
	EXPECT_EQ(std::real(c), 0);
	EXPECT_EQ(std::imag(c), -2);
}

TEST(Complex, operatorDivision0){
	EXPECT_EQ(Complex(14, 6)/2, Complex(7, 3));
}

TEST(Complex, operatorComparison){
	EXPECT_TRUE(Complex(7, 3) == Complex(7, 3));
	EXPECT_FALSE(Complex(7, 3) == Complex(7, 0));
	EXPECT_FALSE(Complex(7, 3) == Complex(0, 3));
}

TEST(Complex, operatorNotEqual){
	EXPECT_FALSE(Complex(7, 3) != Complex(7, 3));
	EXPECT_TRUE(Complex(7, 3) != Complex(7, 0));
	EXPECT_TRUE(Complex(7, 3) != Complex(0, 3));
}

TEST(Complex, operatorOstream){
	Complex complex(7, 3);
	std::stringstream ss;
	ss << complex;

	EXPECT_TRUE(ss.str().compare("(7,3)") == 0);
}

TEST(Complex, operatorIstream){
	Complex complex;
	std::stringstream ss("(7,3)");
	ss >> complex;

	EXPECT_EQ(complex, Complex(7, 3));
}

TEST(Complex, real){
	EXPECT_EQ(real(Complex(7, 3)), 7);
}

TEST(Complex, imag){
	EXPECT_EQ(imag(Complex(7, 3)), 3);
}

TEST(Complex, conj){
	EXPECT_EQ(conj(Complex(7, 3)), Complex(7, -3));
}

TEST(Complex, abs){
	EXPECT_EQ(abs(Complex(3, 4)), 5);
}

#endif

};
