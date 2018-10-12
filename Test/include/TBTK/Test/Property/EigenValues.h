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
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput[n] = n;
	EigenValues eigenValues(1000, dataInput);
	ASSERT_EQ(eigenValues.getSize(), 1000);
	const std::vector<double> &data = eigenValues.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(EigenValues, SerializeToJSON){
	double dataInput[1000];
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
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput0[n] = n;
	EigenValues eigenValues0(1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput1[n] = 2*n;
	EigenValues eigenValues1(1000, dataInput1);

	eigenValues0 += eigenValues1;
	const std::vector<double> &data0 = eigenValues0.getData();
	for(unsigned int n = 0; n < data0.size(); n++)
		EXPECT_DOUBLE_EQ(data0[n], 3*n);
}

TEST(EigenValues, operatorSubtractionAssignment){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput0[n] = n;
	EigenValues eigenValues0(1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n ++)
		dataInput1[n] = 2*n;
	EigenValues eigenValues1(1000, dataInput1);

	eigenValues0 -= eigenValues1;
	const std::vector<double> &data0 = eigenValues0.getData();
	for(int n = 0; n < (int)data0.size(); n++)
		EXPECT_DOUBLE_EQ(data0[n], -n);
}

TEST(EigenValues, serialize){
	//Already tested through SerializeToJSON.
}

};	//End of namespace Property
};	//End of namespace TBTK
