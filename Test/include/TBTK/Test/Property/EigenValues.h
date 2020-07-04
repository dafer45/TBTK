#include "TBTK/Property/EigenValues.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(EigenValues, Constructor0){
	EigenValues eigenValues(1000);
	ASSERT_EQ(eigenValues.getSize(), 1000);
	const std::vector<double> &data = eigenValues.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(EigenValues, Constructor1){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues(1000, dataInput);
	ASSERT_EQ(eigenValues.getSize(), 1000);
	const std::vector<double> &data = eigenValues.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(EigenValues, SerializeToJSON){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues0(1000, dataInput);
	EigenValues eigenValues1(
		eigenValues0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(eigenValues1.getSize(), 1000);
	const std::vector<double> &data = eigenValues1.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(EigenValues, operatorAdditionAssignment){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput0[n] = n;
	EigenValues eigenValues0(1000, dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput1[n] = 2*n;
	EigenValues eigenValues1(1000, dataInput1);

	eigenValues0 += eigenValues1;
	const std::vector<double> &data0 = eigenValues0.getData();
	for(unsigned int n = 0; n < data0.size(); n++)
		EXPECT_DOUBLE_EQ(data0[n], 3*n);
}

TEST(EigenValues, operatorAddition){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput0[n] = n;
	EigenValues eigenValues0(1000, dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput1[n] = 2*n;
	EigenValues eigenValues1(1000, dataInput1);

	EigenValues eigenValues2 = eigenValues0 + eigenValues1;
	const std::vector<double> &data2 = eigenValues2.getData();
	for(unsigned int n = 0; n < data2.size(); n++)
		EXPECT_DOUBLE_EQ(data2[n], 3*n);
}

TEST(EigenValues, operatorSubtractionAssignment){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput0[n] = n;
	EigenValues eigenValues0(1000, dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput1[n] = 2*n;
	EigenValues eigenValues1(1000, dataInput1);

	eigenValues0 -= eigenValues1;
	const std::vector<double> &data0 = eigenValues0.getData();
	for(int n = 0; n < (int)data0.size(); n++)
		EXPECT_DOUBLE_EQ(data0[n], -n);
}

TEST(EigenValues, operatorSubtraction){
	CArray<double> dataInput0(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput0[n] = n;
	EigenValues eigenValues0(1000, dataInput0);

	CArray<double> dataInput1(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput1[n] = 2*n;
	EigenValues eigenValues1(1000, dataInput1);

	EigenValues eigenValues2 = eigenValues0 - eigenValues1;
	const std::vector<double> &data2 = eigenValues2.getData();
	for(int n = 0; n < (int)data2.size(); n++)
		EXPECT_DOUBLE_EQ(data2[n], -n);
}

TEST(EigenValues, operatorMultiplicationAssignment){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues(1000, dataInput);

	eigenValues *= 2;
	const std::vector<double> &data = eigenValues.getData();
	for(int n = 0; n < (int)data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 2*n);
}

TEST(EigenValues, operatorMultiplication){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues0(1000, dataInput);

	EigenValues eigenValues1 = eigenValues0*2;
	EigenValues eigenValues2 = 2*eigenValues0;
	const std::vector<double> &data1 = eigenValues1.getData();
	for(int n = 0; n < (int)data1.size(); n++)
		EXPECT_DOUBLE_EQ(data1[n], 2*n);
	const std::vector<double> &data2 = eigenValues2.getData();
	for(int n = 0; n < (int)data2.size(); n++)
		EXPECT_DOUBLE_EQ(data2[n], 2*n);
}

TEST(EigenValues, operatorDivisionAssignment){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues(1000, dataInput);

	eigenValues /= 2;
	const std::vector<double> &data = eigenValues.getData();
	for(int n = 0; n < (int)data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n/2.);
}

TEST(EigenValues, operatorDivision){
	CArray<double> dataInput(1000);
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues0(1000, dataInput);

	EigenValues eigenValues1 = eigenValues0/2;
	const std::vector<double> &data1 = eigenValues1.getData();
	for(int n = 0; n < (int)data1.size(); n++)
		EXPECT_DOUBLE_EQ(data1[n], n/2.);
}

TEST(EigenValues, serialize){
	//Already tested through SerializeToJSON.
}

};	//End of namespace Property
};	//End of namespace TBTK
