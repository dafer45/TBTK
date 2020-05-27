#include "TBTK/Math/ArrayAlgorithms.h"

#include "gtest/gtest.h"

#include <cmath>

//Note: Math functions are implemented in the TBTK::Math namespace as non-class
//member functions. The ArrayAlgorithm class is a helper class that allows for
//direct access to the Array class members by having friend status. These
//functions are then made available through wrapper functions that makes these
//available as non-class member functions. These tests therefore call the
//wrapper functions rather than the member functions, since these are the ones
//that are supposed to be used everywhere else.

namespace TBTK{
namespace Math{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

class ArrayAlgorithmsTest : public ::testing::Test{
protected:
	Array<double> A, B, C;
	Array<std::complex<double>> D;
	void SetUp() override{
		A = Array<double>({2, 3});
		A[{0, 0}] = 1;
		A[{0, 1}] = 2;
		A[{0, 2}] = 3;
		A[{1, 0}] = 4;
		A[{1, 1}] = 5;
		A[{1, 2}] = 6;

		B = Array<double>({2, 3});
		B[{0, 0}] = 0.1;
		B[{0, 1}] = 0.2;
		B[{0, 2}] = 0.3;
		B[{1, 0}] = 0.4;
		B[{1, 1}] = 0.5;
		B[{1, 2}] = 0.6;

		C = Array<double>({2, 3});
		C[{0, 0}] = 1;
		C[{0, 1}] = 2;
		C[{0, 2}] = 3;
		C[{1, 0}] = -4;
		C[{1, 1}] = -5;
		C[{1, 2}] = -6;

		D = Array<std::complex<double>>({2, 3});
		D[{0, 0}] = std::complex<double>(1, 1);
		D[{0, 1}] = std::complex<double>(-2, 2);
		D[{0, 2}] = std::complex<double>(3, 3);
		D[{1, 0}] = std::complex<double>(-4, -4);
		D[{1, 1}] = std::complex<double>(5, -5);
		D[{1, 2}] = std::complex<double>(-6, -6);
	}

	void verifyArrayDimensions(const Array<double> &E){
		const std::vector<unsigned int> &ranges = E.getRanges();
		EXPECT_EQ(ranges.size(), 2);
		EXPECT_EQ(ranges[0], 2);
		EXPECT_EQ(ranges[1], 3);
	}

	void verifyArrayDimensions(const Array<std::complex<double>> &E){
		const std::vector<unsigned int> &ranges = E.getRanges();
		EXPECT_EQ(ranges.size(), 2);
		EXPECT_EQ(ranges[0], 2);
		EXPECT_EQ(ranges[1], 3);
	}
};

//TBTKFeature Math.ArrayAlgorithms.sin.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, sin0){
	Array<double> result = sin(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::sin(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::sin(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::sin(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::sin(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::sin(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::sin(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.cos.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, cos0){
	Array<double> result = cos(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::cos(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::cos(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::cos(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::cos(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::cos(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::cos(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.tan.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, tan0){
	Array<double> result = tan(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::tan(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::tan(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::tan(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::tan(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::tan(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::tan(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.asin.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, asin0){
	Array<double> result = asin(B);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::asin(0.1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::asin(0.2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::asin(0.3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::asin(0.4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::asin(0.5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::asin(0.6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.acos.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, acos0){
	Array<double> result = acos(B);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::acos(0.1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::acos(0.2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::acos(0.3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::acos(0.4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::acos(0.5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::acos(0.6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.atan.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, atan0){
	Array<double> result = atan(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::atan(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::atan(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::atan(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::atan(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::atan(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::atan(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.sinh.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, sinh0){
	Array<double> result = sinh(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::sinh(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::sinh(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::sinh(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::sinh(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::sinh(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::sinh(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.cosh.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, cosh0){
	Array<double> result = cosh(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::cosh(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::cosh(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::cosh(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::cosh(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::cosh(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::cosh(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.tanh.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, tanh0){
	Array<double> result = tanh(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::tanh(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::tanh(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::tanh(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::tanh(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::tanh(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::tanh(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.asinh.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, asinh0){
	Array<double> result = asinh(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::asinh(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::asinh(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::asinh(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::asinh(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::asinh(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::asinh(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.acosh.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, acosh0){
	Array<double> result = acosh(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::acosh(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::acosh(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::acosh(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::acosh(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::acosh(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::acosh(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.atanh.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, atanh0){
	Array<double> result = atanh(B);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::atanh(0.1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::atanh(0.2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::atanh(0.3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::atanh(0.4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::atanh(0.5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::atanh(0.6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.log.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, log0){
	Array<double> result = log(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::log(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::log(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::log(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::log(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::log(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::log(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.log2.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, log20){
	Array<double> result = log2(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::log2(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::log2(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::log2(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::log2(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::log2(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::log2(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.log10.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, log100){
	Array<double> result = log10(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::log10(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::log10(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::log10(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::log10(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::log10(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::log10(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.pow.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, pow0){
	Array<double> result = pow(A, 3.5);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::pow(1, 3.5), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::pow(2, 3.5), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::pow(3, 3.5), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::pow(4, 3.5), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::pow(5, 3.5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::pow(6, 3.5), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.exp.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, exp0){
	Array<double> result = exp(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::exp(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::exp(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::exp(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::exp(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::exp(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::exp(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.abs.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, abs0){
	Array<double> result = abs(C);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::abs(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::abs(-2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::abs(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::abs(-4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::abs(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::abs(-6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.abs.1 2020-05-18
TEST_F(ArrayAlgorithmsTest, abs1){
	Array<double> result = abs(D);
	verifyArrayDimensions(result);
	EXPECT_NEAR(
		(result[{0, 0}]),
		std::abs(std::complex<double>(1, 1)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 1}]),
		std::abs(std::complex<double>(-2, 2)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 2}]),
		std::abs(std::complex<double>(3, 3)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 0}]),
		std::abs(std::complex<double>(-4, -4)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 1}]),
		std::abs(std::complex<double>(5, -5)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 2}]),
		std::abs(std::complex<double>(-6, -6)),
		EPSILON_100
	);
}

//TBTKFeature Math.ArrayAlgorithms.arg.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, arg0){
	Array<double> result = arg(D);
	verifyArrayDimensions(result);
	EXPECT_NEAR(
		(result[{0, 0}]),
		std::arg(std::complex<double>(1, 1)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 1}]),
		std::arg(std::complex<double>(-2, 2)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 2}]),
		std::arg(std::complex<double>(3, 3)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 0}]),
		std::arg(std::complex<double>(-4, -4)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 1}]),
		std::arg(std::complex<double>(5, -5)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 2}]),
		std::arg(std::complex<double>(-6, -6)),
		EPSILON_100
	);
}

//TBTKFeature Math.ArrayAlgorithms.real.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, real0){
	Array<double> result = real(D);
	verifyArrayDimensions(result);
	EXPECT_NEAR(
		(result[{0, 0}]),
		std::real(std::complex<double>(1, 1)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 1}]),
		std::real(std::complex<double>(-2, 2)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 2}]),
		std::real(std::complex<double>(3, 3)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 0}]),
		std::real(std::complex<double>(-4, -4)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 1}]),
		std::real(std::complex<double>(5, -5)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 2}]),
		std::real(std::complex<double>(-6, -6)),
		EPSILON_100
	);
}

//TBTKFeature Math.ArrayAlgorithms.imag.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, imag0){
	Array<double> result = imag(D);
	verifyArrayDimensions(result);
	EXPECT_NEAR(
		(result[{0, 0}]),
		std::imag(std::complex<double>(1, 1)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 1}]),
		std::imag(std::complex<double>(-2, 2)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{0, 2}]),
		std::imag(std::complex<double>(3, 3)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 0}]),
		std::imag(std::complex<double>(-4, -4)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 1}]),
		std::imag(std::complex<double>(5, -5)),
		EPSILON_100
	);
	EXPECT_NEAR(
		(result[{1, 2}]),
		std::imag(std::complex<double>(-6, -6)),
		EPSILON_100
	);
}

//TBTKFeature Math.ArrayAlgorithms.conj.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, conj0){
	Array<std::complex<double>> result = conj(D);
	verifyArrayDimensions(result);
	EXPECT_NEAR(
		std::real(result[{0, 0}]),
		std::real(std::conj(std::complex<double>(1, 1))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::real(result[{0, 1}]),
		std::real(std::conj(std::complex<double>(-2, 2))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::real(result[{0, 2}]),
		std::real(std::conj(std::complex<double>(3, 3))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::real(result[{1, 0}]),
		std::real(std::conj(std::complex<double>(-4, -4))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::real(result[{1, 1}]),
		std::real(std::conj(std::complex<double>(5, -5))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::real(result[{1, 2}]),
		std::real(std::conj(std::complex<double>(-6, -6))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::imag(result[{0, 0}]),
		std::imag(std::conj(std::complex<double>(1, 1))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::imag(result[{0, 1}]),
		std::imag(std::conj(std::complex<double>(-2, 2))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::imag(result[{0, 2}]),
		std::imag(std::conj(std::complex<double>(3, 3))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::imag(result[{1, 0}]),
		std::imag(std::conj(std::complex<double>(-4, -4))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::imag(result[{1, 1}]),
		std::imag(std::conj(std::complex<double>(5, -5))),
		EPSILON_100
	);
	EXPECT_NEAR(
		std::imag(result[{1, 2}]),
		std::imag(std::conj(std::complex<double>(-6, -6))),
		EPSILON_100
	);
}

//TBTKFeature Math.ArrayAlgorithms.sqrt.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, sqrt0){
	Array<double> result = sqrt(A);
	verifyArrayDimensions(result);
	EXPECT_NEAR((result[{0, 0}]), std::sqrt(1), EPSILON_100);
	EXPECT_NEAR((result[{0, 1}]), std::sqrt(2), EPSILON_100);
	EXPECT_NEAR((result[{0, 2}]), std::sqrt(3), EPSILON_100);
	EXPECT_NEAR((result[{1, 0}]), std::sqrt(4), EPSILON_100);
	EXPECT_NEAR((result[{1, 1}]), std::sqrt(5), EPSILON_100);
	EXPECT_NEAR((result[{1, 2}]), std::sqrt(6), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.max.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, max0){
	EXPECT_EQ(max(A), 6);
}

//TBTKFeature Math.ArrayAlgorithms.min.0 2020-05-18
TEST_F(ArrayAlgorithmsTest, min0){
	EXPECT_EQ(min(A), 1);
}

//TBTKFeature Math.ArrayAlgorithms.trace.0 2020-05-26
TEST(ArrayAlgorithms, trace0){
	Array<unsigned int> array({3, 3});
	for(unsigned int row = 0; row < 3; row++)
		for(unsigned int column = 0; column < 3; column++)
			array[{row, column}] = 3*row + column;

	EXPECT_EQ(trace(array), 12);
}

//TBTKFeature Math.ArrayAlgorithms.trace.1 2020-05-26
TEST(ArrayAlgorithms, trace1){
	Array<unsigned int> array({3, 2});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			trace(array);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Math.ArrayAlgorithms.trace.2 2020-05-26
TEST(ArrayAlgorithms, trace2){
	Array<unsigned int> array({3, 3, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			trace(array);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Math.ArrayAlgorithms.norm.0 2020-05-26
TEST_F(ArrayAlgorithmsTest, norm0){
	EXPECT_NEAR(norm(C), std::sqrt(1 + 4 + 9 + 16 + 25 + 36), EPSILON_100);
}

//TBTKFeature Math.ArrayAlgorithms.norm.1 2020-05-26
TEST_F(ArrayAlgorithmsTest, norm1){
	double reference = 0;
	for(unsigned int n = 1; n < 7; n++)
		reference += std::pow(n, 3.5);
	reference = std::pow(reference, 1/3.5);

	EXPECT_NEAR(norm(C, 3.5), reference, EPSILON_100);
}

};	//End of namespace Math
};	//End of namespace TBTK
