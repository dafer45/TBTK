#include "TBTK/AnnotatedArray.h"
#include "TBTK/Array.h"

#include "gtest/gtest.h"

#include <vector>

namespace TBTK{

class AnnotatedArrayTest : public ::testing::Test{
protected:
	Array<unsigned int> array;
	std::vector<std::vector<double>> axes;
	AnnotatedArray<unsigned int, double> annotatedArray;
	void SetUp() override{
		array = Array<unsigned int>({2, 3, 4});
		for(unsigned int x = 0; x < 2; x++)
			for(unsigned int y = 0; y < 3; y++)
				for(unsigned int z = 0; z < 4; z++)
					array[{x, y, z}] = x*y*z;

		axes = std::vector<std::vector<double>>(3);
		for(unsigned int x = 0; x < 2; x++)
			axes[0].push_back(x);
		for(unsigned int y = 0; y < 3; y++)
			axes[1].push_back(y);
		for(unsigned int z = 0; z < 4; z++)
			axes[2].push_back(z);

		annotatedArray
			= AnnotatedArray<unsigned int, double>(array, axes);
	}
};

//TBTKFeature Utilities.AnnotatedArray.construction.1 2019-11-26
TEST(AnnotatedArray, constructor1){
	Array<unsigned int> array({2, 3, 4});
	std::vector<std::vector<double>> axes
		= std::vector<std::vector<double>>(2);
	for(unsigned int x = 0; x < 2; x++)
		axes[0].push_back(x);
	for(unsigned int y = 0; y < 3; y++)
		axes[1].push_back(y);

	EXPECT_EXIT(
		({
			Streams::setStdMuteErr();
			AnnotatedArray<unsigned int, double> annotatedarray(
				array,
				axes
			);
		}),
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.AnnotatedArray.construction.2 2019-11-26
TEST(AnnotatedArray, constructor2){
	Array<unsigned int> array({2, 3, 4});
	std::vector<std::vector<double>> axes
		= std::vector<std::vector<double>>(3);
	for(unsigned int x = 0; x < 2; x++)
		axes[0].push_back(x);
	for(unsigned int y = 0; y < 3; y++)
		axes[1].push_back(y);
	for(unsigned int z = 0; z < 3; z++)
		axes[2].push_back(z);

	EXPECT_EXIT(
		({
			Streams::setStdMuteErr();
			AnnotatedArray<unsigned int, double> annotatedarray(
				array,
				axes
			);
		}),
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.AnnotatedArray.construction.3 2019-11-26
TEST_F(AnnotatedArrayTest, constructor3){
	EXPECT_EQ(annotatedArray, array);
	const std::vector<std::vector<double>> annotatedArrayAxes
		= annotatedArray.getAxes();
	EXPECT_EQ(axes.size(), annotatedArrayAxes.size());
	for(unsigned int n = 0; n < axes.size(); n++){
		EXPECT_EQ(axes[n].size(), annotatedArrayAxes[n].size());
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_FLOAT_EQ(axes[n][c], annotatedArrayAxes[n][c]);
	}
}

};
