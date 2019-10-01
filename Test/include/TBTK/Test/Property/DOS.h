#include "TBTK/Property/DOS.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(DOS, Constructor0){
	DOS dos(-10, 10, 1000);
	EXPECT_EQ(dos.getLowerBound(), -10);
	EXPECT_EQ(dos.getUpperBound(), 10);
	ASSERT_EQ(dos.getResolution(), 1000);
	const std::vector<double> &data = dos.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], 0);
}

TEST(DOS, Constructor1){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	DOS dos(-10, 10, 1000, dataInput);
	EXPECT_EQ(dos.getLowerBound(), -10);
	EXPECT_EQ(dos.getUpperBound(), 10);
	ASSERT_EQ(dos.getResolution(), 1000);
	const std::vector<double> &data = dos.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(DOS, SerializeToJSON){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	DOS dos0(-10, 10, 1000, dataInput);
	DOS dos1(
		dos0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(dos1.getLowerBound(), -10);
	EXPECT_EQ(dos1.getUpperBound(), 10);
	ASSERT_EQ(dos1.getResolution(), 1000);
	const std::vector<double> &data = dos1.getData();
	for(unsigned int n = 0; n < data.size(); n++)
		EXPECT_DOUBLE_EQ(data[n], n);
}

TEST(DOS, operatorAdditionAssignment){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	DOS dos0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	DOS dos1(-10, 10, 1000, dataInput1);

	dos0 += dos1;
	const std::vector<double> &data0 = dos0.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], 3*n);
}

TEST(DOS, operatorAddition){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	DOS dos0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	DOS dos1(-10, 10, 1000, dataInput1);

	DOS dos2 = dos0 + dos1;
	const std::vector<double> &data2 = dos2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 3*n);
}

TEST(DOS, operatorSubtractionAssignment){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	DOS dos0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	DOS dos1(-10, 10, 1000, dataInput1);

	dos0 -= dos1;
	const std::vector<double> &data0 = dos0.getData();
	for(int n = 0; n < 1000; n++)
		EXPECT_EQ(data0[n], -n);
}

TEST(DOS, operatorSubtraction){
	double dataInput0[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput0[n] = n;
	DOS dos0(-10, 10, 1000, dataInput0);

	double dataInput1[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput1[n] = 2*n;
	DOS dos1(-10, 10, 1000, dataInput1);

	DOS dos2 = dos0 - dos1;
	const std::vector<double> &data2 = dos2.getData();
	for(int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], -n);
}

TEST(DOS, operatorMultiplicationAssignment){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	DOS dos(-10, 10, 1000, dataInput);

	dos *= 2;
	const std::vector<double> &data = dos.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data[n], 2*n);
}

TEST(DOS, operatorMultiplication){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	DOS dos0(-10, 10, 1000, dataInput);

	DOS dos1 = dos0*2;
	DOS dos2 = 2*dos0;
	const std::vector<double> &data1 = dos1.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data1[n], 2*n);
	const std::vector<double> &data2 = dos2.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data2[n], 2*n);
}

TEST(DOS, operatorDivisionAssignment){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	DOS dos(-10, 10, 1000, dataInput);

	dos /= 2;
	const std::vector<double> &data = dos.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data[n], n/2.);
}

TEST(DOS, operatorDivision){
	double dataInput[1000];
	for(unsigned int n = 0; n < 1000; n++)
		dataInput[n] = n;
	DOS dos0(-10, 10, 1000, dataInput);

	DOS dos1 = dos0/2;
	const std::vector<double> &data1 = dos1.getData();
	for(unsigned int n = 0; n < 1000; n++)
		EXPECT_EQ(data1[n], n/2.);
}

TEST(DOS, toString){
	//Difficult to formulate a good test criterium for.
}

TEST(DOS, serialize){
	//Already tested through SerializeToJSON.
}

};	//End of namespace Property
};	//End of namespace TBTK
