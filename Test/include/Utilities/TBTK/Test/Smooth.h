#include "TBTK/Smooth.h"

#include "gtest/gtest.h"

namespace TBTK{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

class SmoothTest : public ::testing::Test{
protected:
	const unsigned int DATA_SIZE = 100;
	const unsigned int GAUSSIAN_WINDOW_SIZE = 21;
	const double GAUSSIAN_SIGMA = 3;
	CArray<double> gaussianInput;
	CArray<double> gaussianReference;
	void SetUp() override{
		gaussianInput = CArray<double>(DATA_SIZE);
		for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
			gaussianInput[n] = n;

		double normalization = 0;
		for(
			int n = -(int)GAUSSIAN_WINDOW_SIZE/2;
			n <= (int)GAUSSIAN_WINDOW_SIZE/2;
			n++
		){
			normalization += exp(-n*n/(2*pow(GAUSSIAN_SIGMA, 2)));
		}
		normalization = 1/normalization;

		gaussianReference = CArray<double>(DATA_SIZE);
		for(int n = 0; n < (int)gaussianInput.getSize(); n++){
			gaussianReference[n] = 0;
			for(
				int c = std::max(
					0,
					(int)n
					- (int)GAUSSIAN_WINDOW_SIZE/2
				);
				c < std::min(
					(int)n
					+ (int)GAUSSIAN_WINDOW_SIZE/2
					+ 1,
					(int)gaussianInput.getSize()
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
TEST_F(SmoothTest, gaussian0){
	Array<double> input({gaussianInput.getSize()});
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		input[{n}] = gaussianInput[n];

	Array<double> result = Smooth::gaussian(
		input,
		GAUSSIAN_SIGMA,
		GAUSSIAN_WINDOW_SIZE
	);
	EXPECT_EQ(result.getSize(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result[{n}], gaussianReference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.1 2020-07-04
TEST_F(SmoothTest, gaussian1){
	std::vector<double> input;
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		input.push_back(gaussianInput[n]);

	std::vector<double> result = Smooth::gaussian(
		input,
		GAUSSIAN_SIGMA,
		GAUSSIAN_WINDOW_SIZE
	);
	EXPECT_EQ(result.size(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.size(); n++)
		EXPECT_NEAR(result[n], gaussianReference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.2 2020-07-04
TEST_F(SmoothTest, gaussian2){
	CArray<double> input(gaussianInput.getSize());
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		input[n] = gaussianInput[n];

	CArray<double> result = Smooth::gaussian(
		input,
		GAUSSIAN_SIGMA,
		GAUSSIAN_WINDOW_SIZE
	);
	EXPECT_EQ(result.getSize(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result[n], gaussianReference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.3 2020-07-04
TEST_F(SmoothTest, gaussian3){
	CArray<double> data(gaussianInput.getSize());
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		data[n] = gaussianInput[n];
	Property::DOS inputDOS(Range(-1, 1, data.getSize()), data);

	Property::DOS result = Smooth::gaussian(
		inputDOS,
		GAUSSIAN_SIGMA*(
			inputDOS.getUpperBound() - inputDOS.getLowerBound()
		)/inputDOS.getResolution(),
		GAUSSIAN_WINDOW_SIZE
	);
	EXPECT_EQ(result.getSize(), gaussianReference.getSize());
	for(unsigned int n = 0; n < result.getSize(); n++)
		EXPECT_NEAR(result(n), gaussianReference[n], EPSILON_100);
}

//TBTKFeature Utilities.Smooth.gaussian.4 2020-07-04
TEST_F(SmoothTest, gaussian4){
	IndexTree indexTree;
	indexTree.add({0, 1});
	indexTree.add({1, 2});
	indexTree.add({3});
	indexTree.generateLinearMap();

	CArray<double> data(3*gaussianInput.getSize());
	for(unsigned int n = 0; n < gaussianInput.getSize(); n++)
		for(unsigned int c = 0; c < 3; c++)
			data[c*gaussianInput.getSize() + n] = gaussianInput[n];

	Property::LDOS inputLDOS(
		indexTree,
		Range(-1, 1, gaussianInput.getSize()),
		data
	);

	Property::LDOS result = Smooth::gaussian(
		inputLDOS,
		GAUSSIAN_SIGMA*(
			inputLDOS.getUpperBound() - inputLDOS.getLowerBound()
		)/inputLDOS.getResolution(),
		GAUSSIAN_WINDOW_SIZE
	);
	EXPECT_EQ(result.getSize(), 3*gaussianReference.getSize());
	for(unsigned int n = 0; n < DATA_SIZE; n++){
		EXPECT_NEAR(result({0, 1}, n), gaussianReference[n], EPSILON_100);
		EXPECT_NEAR(result({1, 2}, n), gaussianReference[n], EPSILON_100);
		EXPECT_NEAR(result({3}, n), gaussianReference[n], EPSILON_100);
	}
}

//TBTKFeature Utilities.Smooth.gaussian.5 2020-07-04
TEST_F(SmoothTest, gaussian5){
	//TODO: Implement test for SpinPolarizedLDOS.
}

};
