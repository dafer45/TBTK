#include "TBTK/Smooth.h"

#include "gtest/gtest.h"

namespace TBTK{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

class SmoothTest : public ::testing::Test{
protected:
	const unsigned int DATA_SIZE;
	const unsigned int GAUSSIAN_WINDOW_SIZE = 21;
	const double GAUSSIAN_SIGMA = 3;
	const un
	CArray<double> gausianInput;
	CArray<double> gausianReference;
	void SetUp() override{
		gaussianInput = CArray<double>(DATA_SIZE);
		gaussianReference = CArray<double>(DATA_SIZE, 0);
		for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
			gaussianInput[n] = n;

		double normalization = 0;
		for(
			int n = -GAUSSIAN_WINDOW_SIZE/2;
			n <= GAUSSIAN_WINDOW_SIZE/2;
			n++
		){
			normalization += exp(-n*n/(2*pow(GAUSSIAN_SIGMA, 2)));
		}
		normalization = 1/normalization;

		for(int n = 0; n < (int)gaussianInput.getSize(); n++){
			for(
				int c = std::max(
					0,
					(int)n
					- (int)GAUSSIAN_WINDOW_SIZE/2
					+ 1
				);
				c < std::min(
					(int)n
					+ (int)GAUSSIAN_WINDOW_SIZE/2
					+ 1,
					gaussianInput.getSize()
				);
				c++
			){
				gaussianReference[n] += gaussianInput[c]*exp(
					-(c-n)*(c-n)/(2*pow(GAUSSIAN_SIGMA, 2))
				);
			}
			gaussianReference[n] *= normalization;
		}
	}
};

//TBTKFeature Utilities.Smooth.gaussian.0 2020-07-04
TEST(CArray, gaussian0){
	Array<double> input({gaussianInput.getSize()});
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		input[{n}] = gaussianInput[n];

	Array<double> result = Smooth::gaussian(input);
	EXPECT_EQ(result.getSize(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result[{n}], reference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.1 2020-07-04
TEST(CArray, gaussian1){
	std::vector<double> input;
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		input.push_back(gaussianInput[n]);

	std::vector<double> result = Smooth::gaussian(input);
	EXPECT_EQ(result.size(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result[n], reference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.2 2020-07-04
TEST(CArray, gaussian2){
	CArray<double> input(gaussianInput.getSize());
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		input[n] = gaussianInput[n];

	CArray<double> result = Smooth::gaussian(input);
	EXPECT_EQ(result.getSize(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result[n], reference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.3 2020-07-04
TEST(CArray, gaussian3){
	CArray<double> data(gaussianInput.getSize());
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		data[n] = gaussianInput[n];
	Property::DOS inputDOS(Range(-1, 1, data.getSize()), data)

	Property::DOS result = Smooth::gaussian(inputDOS);
	EXPECT_EQ(result.getSize(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result(n), reference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.4 2020-07-04
TEST(CArray, gaussian4){
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({3});
	indexTree.generateLinearMap();

	CArray<double> data(3*gaussianInput.getSize());
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		for(unsigned int c = 0; c < 3; c++)
			data[c*gaussianInput.getSize() + n] = gaussianInput[n];

	Property::DOS inputLDOS(indexTree, Range(-1, 1, data.getSize()), data)

	Property::DOS result = Smooth::gaussian(inputLDOS);
	EXPECT_EQ(result.getSize(), 3*gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++){
		EXPECT_NEAR(result({0, 1}, n), reference[n], EPSILON_100);
		EXPECT_NEAR(result({1, 2}, n), reference[n], EPSILON_100);
		EXPECT_NEAR(result({3}, n), reference[n], EPSILON_100);
	}
}

//TBTKFeature Utilities.Smooth.gaussian.5 2020-07-04
TEST(CArray, gaussian5){
	//TODO: Implement test for SpinPolarizedLDOS.
}

};
