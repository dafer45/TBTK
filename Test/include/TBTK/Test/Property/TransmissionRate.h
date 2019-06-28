#include "TBTK/Property/TransmissionRate.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(TransmissionRate, Constructor0){
	TransmissionRate transmissionRate(-10, 10, 1000);
	EXPECT_EQ(transmissionRate.getLowerBound(), -10);
	EXPECT_EQ(transmissionRate.getUpperBound(), 10);
	ASSERT_EQ(transmissionRate.getResolution(), 1000);
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(TransmissionRate, Constructor1){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate(-10, 10, 1000, dataInput);
	EXPECT_EQ(transmissionRate.getLowerBound(), -10);
	EXPECT_EQ(transmissionRate.getUpperBound(), 10);
	ASSERT_EQ(transmissionRate.getResolution(), 1000);
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(TransmissionRate, SerializeToJSON){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput);
	TransmissionRate transmissionRate1(
		transmissionRate0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(transmissionRate1.getLowerBound(), -10);
	EXPECT_EQ(transmissionRate1.getUpperBound(), 10);
	ASSERT_EQ(transmissionRate1.getResolution(), 1000);
	const std::vector<double> &data = transmissionRate1.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(TransmissionRate, operatorAdditionAssignment){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(-10, 10, 1000, dataInput1);

	transmissionRate0 += transmissionRate1;
	const std::vector<double> &data0 = transmissionRate0.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], 3*n);
}

TEST(TransmissionRate, operatorAddition){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(-10, 10, 1000, dataInput1);

	TransmissionRate transmissionRate2
		= transmissionRate0 + transmissionRate1;
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 3*n);
}

TEST(TransmissionRate, operatorSubtractionAssignment){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(-10, 10, 1000, dataInput1);

	transmissionRate0 -= transmissionRate1;
	const std::vector<double> &data0 = transmissionRate0.getData();
	for(int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], -n);
}

TEST(TransmissionRate, operatorSubtraction){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	TransmissionRate transmissionRate1(-10, 10, 1000, dataInput1);

	TransmissionRate transmissionRate2
		= transmissionRate0 - transmissionRate1;
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], -n);
}

TEST(TransmissionRate, operatorMultiplicationAssignment){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate(-10, 10, 1000, dataInput);

	transmissionRate *= 2;
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data[n], 2*n);
}

TEST(TransmissionRate, operatorMultiplication){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput);

	TransmissionRate transmissionRate1 = transmissionRate0*2;
	TransmissionRate transmissionRate2 = 2*transmissionRate0;
	const std::vector<double> &data1 = transmissionRate1.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data1[n], 2*n);
	const std::vector<double> &data2 = transmissionRate2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 2*n);
}

TEST(TransmissionRate, operatorDivisionAssignment){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate(-10, 10, 1000, dataInput);

	transmissionRate /= 2;
	const std::vector<double> &data = transmissionRate.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data[n], n/2.);
}

TEST(TransmissionRate, operatorDivision){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	TransmissionRate transmissionRate0(-10, 10, 1000, dataInput);

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
