#include "TBTK/Property/TransmissionRate.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

TEST(TransmissionRate, Constructor0){
	TransmissionRate transmissionRate(Range(-10, 10, 1000));
	EXPECT_NEAR(transmissionRate.getLowerBound(), -10, EPSILON_100);
	EXPECT_NEAR(transmissionRate.getUpperBound(), 10, EPSILON_100);
	ASSERT_EQ(transmissionRate.getResolution(), 1000);
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(TransmissionRate, Constructor1){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate(Range(-10, 10, 1000), dataInput);
	EXPECT_NEAR(transmissionRate.getLowerBound(), -10, EPSILON_100);
	EXPECT_NEAR(transmissionRate.getUpperBound(), 10, EPSILON_100);
	ASSERT_EQ(transmissionRate.getResolution(), 1000);
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(TransmissionRate, SerializeToJSON){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput);
	TransmissionRate transmissionRate1(
		transmissionRate0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_NEAR(transmissionRate1.getLowerBound(), -10, EPSILON_100);
	EXPECT_NEAR(transmissionRate1.getUpperBound(), 10, EPSILON_100);
	ASSERT_EQ(transmissionRate1.getResolution(), 1000);
	const std::vector<double> &data = transmissionRate1.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(TransmissionRate, operatorAdditionAssignment){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	transmissionRate0 += transmissionRate1;
	const std::vector<double> &data0 = transmissionRate0.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], 3*n);
}

TEST(TransmissionRate, operatorAddition){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	TransmissionRate transmissionRate2
		= transmissionRate0 + transmissionRate1;
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 3*n);
}

TEST(TransmissionRate, operatorSubtractionAssignment){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	transmissionRate0 -= transmissionRate1;
	const std::vector<double> &data0 = transmissionRate0.getData();
	for(int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], -n);
}

TEST(TransmissionRate, operatorSubtraction){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	TransmissionRate transmissionRate2
		= transmissionRate0 - transmissionRate1;
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], -n);
}

TEST(TransmissionRate, operatorMultiplicationAssignment0){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	transmissionRate0 *= transmissionRate1;
	const std::vector<double> &data0 = transmissionRate0.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], 2*n*n);

	CArray<double> dataInput2(999);
	for(unsigned int n = 0; n < 999; n++)
		dataInput2[n] = 2*n;
	TransmissionRate transmissionRate2(Range(-10, 10, 999), dataInput2);

	TransmissionRate transmissionRate3(Range(-11, 10, 999), dataInput0);
	TransmissionRate transmissionRate4(Range(-10, 11, 999), dataInput0);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			transmissionRate0 *= transmissionRate2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			transmissionRate0 *= transmissionRate3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			transmissionRate0 *= transmissionRate4;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(TransmissionRate, operatorMultiplication0){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	TransmissionRate transmissionRate2
		= transmissionRate0*transmissionRate1;
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 2*n*n);

	CArray<double> dataInput3(999);
	for(unsigned int n = 0; n < 999; n++)
		dataInput3[n] = 2*n;
	TransmissionRate transmissionRate3(Range(-10, 10, 999), dataInput3);

	TransmissionRate transmissionRate4(Range(-11, 10, 999), dataInput0);
	TransmissionRate transmissionRate5(Range(-10, 11, 999), dataInput0);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			TransmissionRate transmissionRate
				= transmissionRate0*transmissionRate3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			TransmissionRate transmissionRate
				= transmissionRate0*transmissionRate4;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			TransmissionRate transmissionRate
				= transmissionRate0*transmissionRate5;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(TransmissionRate, operatorMultiplicationAssignment1){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate(Range(-10, 10, 1000), dataInput);

	transmissionRate *= 2;
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data[n], 2*n);
}

TEST(TransmissionRate, operatorMultiplication1){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput);

	TransmissionRate transmissionRate1 = transmissionRate0*2;
	TransmissionRate transmissionRate2 = 2*transmissionRate0;
	const std::vector<double> &data1 = transmissionRate1.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data1[n], 2*n);
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 2*n);
}

TEST(TransmissionRate, operatorDivisionAssignment0){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 1 + 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	transmissionRate0 /= transmissionRate1;
	const std::vector<double> &data0 = transmissionRate0.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_DOUBLE_EQ(data0[n], n/(1 + 2.*n));

	CArray<double> dataInput2(999);
	for(unsigned int n = 0; n < 999; n++)
		dataInput2[n] = 1 + 2*n;
	TransmissionRate transmissionRate2(Range(-10, 10, 999), dataInput2);

	TransmissionRate transmissionRate3(Range(-11, 10, 999), dataInput0);
	TransmissionRate transmissionRate4(Range(-10, 11, 999), dataInput0);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			transmissionRate0 /= transmissionRate2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			transmissionRate0 /= transmissionRate3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			transmissionRate0 /= transmissionRate4;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(TransmissionRate, operatorDivision0){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 1 + 2*n;
	TransmissionRate transmissionRate1(Range(-10, 10, 1000), dataInput1);

	TransmissionRate transmissionRate2
		= transmissionRate0/transmissionRate1;
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_DOUBLE_EQ(data2[n], n/(1 + 2.*n));

	CArray<double> dataInput3(999);
	for(unsigned int n = 0; n < 999; n++)
		dataInput3[n] = 1 + 2*n;
	TransmissionRate transmissionRate3(Range(-10, 10, 999), dataInput3);

	TransmissionRate transmissionRate4(Range(-11, 10, 999), dataInput0);
	TransmissionRate transmissionRate5(Range(-10, 11, 999), dataInput0);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			TransmissionRate transmissionRate
				= transmissionRate0/transmissionRate3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			TransmissionRate transmissionRate
				= transmissionRate0/transmissionRate4;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			TransmissionRate transmissionRate
				= transmissionRate0/transmissionRate5;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(TransmissionRate, operatorDivisionAssignment1){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate(Range(-10, 10, 1000), dataInput);

	transmissionRate /= 2;
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data[n], n/2.);
}

TEST(TransmissionRate, operatorDivision1){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate0(Range(-10, 10, 1000), dataInput);

	TransmissionRate transmissionRate1 = transmissionRate0/2;
	const std::vector<double> &data1 = transmissionRate1.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data1[n], n/2.);
}

TEST(TransmissionRate, serialize){
	//Already tested through SerializeToJSON.
}

};	//End of namespace Property
};	//End of namespace TBTK
