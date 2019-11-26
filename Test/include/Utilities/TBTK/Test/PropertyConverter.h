#include "TBTK/Property/Density.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/PropertyConverter.h"

#include "gtest/gtest.h"

namespace TBTK{

class PropertyConverterTest : public ::testing::Test{
protected:
	double ENERGY_LOWER_BOUND;
	double ENERGY_UPPER_BOUND;
	int ENERGY_RESOLUTION;
	unsigned int RANGES_SIZE_X;
	unsigned int RANGES_SIZE_Y;

	Property::DOS dos;
	Property::Density densityRanges;
	Property::Density densityCustom;
	Property::LDOS ldosRanges;

	void SetUp() override{
		ENERGY_LOWER_BOUND = -10;
		ENERGY_UPPER_BOUND = 10;
		ENERGY_RESOLUTION = 10;
		RANGES_SIZE_X = 2;
		RANGES_SIZE_Y = 3;

		SetUpDOS();
		SetUpDensityRanges();
		SetUpDensityCustom();
		SetUpLDOSRanges();
	}

	void SetUpDOS(){
		dos = Property::DOS(
			ENERGY_LOWER_BOUND,
			ENERGY_UPPER_BOUND,
			ENERGY_RESOLUTION
		);
		for(unsigned int n = 0; n < 10; n++)
			dos(n) = n;
	}

	void SetUpDensityRanges(){
		densityRanges = Property::Density(
			{(int)RANGES_SIZE_X, (int)RANGES_SIZE_Y}
		);
		std::vector<double> &data = densityRanges.getDataRW();
		for(unsigned int x = 0; x < RANGES_SIZE_X; x++){
			for(unsigned int y = 0; y < RANGES_SIZE_Y; y++){
				for(
					unsigned int e = 0;
					e < (unsigned int)ENERGY_RESOLUTION;
					e++
				){
					data[RANGES_SIZE_Y*x + y] = x*y;
				}
			}
		}
	}

	void SetUpDensityCustom(){
		IndexTree indexTree;
		indexTree.add({1, 4});
		indexTree.add({2, 2});
		indexTree.add({2, 3});
		indexTree.generateLinearMap();
		densityCustom = Property::Density(indexTree);

		densityCustom({1, 4}) = 1;
		densityCustom({2, 2}) = 2;
		densityCustom({2, 3}) = 3;
	}

	void SetUpLDOSRanges(){
		ldosRanges = Property::LDOS(
			{(int)RANGES_SIZE_X, (int)RANGES_SIZE_Y},
			ENERGY_LOWER_BOUND,
			ENERGY_UPPER_BOUND,
			ENERGY_RESOLUTION
		);
		std::vector<double> &data = ldosRanges.getDataRW();
		for(unsigned int x = 0; x < RANGES_SIZE_X; x++){
			for(unsigned int y = 0; y < RANGES_SIZE_Y; y++){
				for(
					unsigned int e = 0;
					e < (unsigned int)ENERGY_RESOLUTION;
					e++
				){
					data[
						ENERGY_RESOLUTION*(
							RANGES_SIZE_Y*x + y
						) + e
					] = x*y*e;
				}
			}
		}
	}
};

//TBTKFeature Utilities.PropertyConverter.convert.1 2019-11-26
TEST_F(PropertyConverterTest, convert1){
	AnnotatedArray<double, Subindex> result
		= PropertyConverter::convert(dos);

	const std::vector<unsigned int> &ranges = result.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], 10);
	for(unsigned int n = 0; n < ranges[0]; n++)
		EXPECT_EQ(result[{n}], n);

	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(axes.size(), 1);
	for(unsigned int n = 0; n < axes[0].size(); n++)
		EXPECT_EQ(axes[0][n], n);
}

//TBTKFeature Utilities.PropertyConverter.convert.2 2019-11-26
TEST_F(PropertyConverterTest, convert2){
	AnnotatedArray<double, Subindex> result
		= PropertyConverter::convert(densityRanges);

	const std::vector<unsigned int> &ranges = result.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], RANGES_SIZE_X);
	EXPECT_EQ(ranges[1], RANGES_SIZE_Y);
	for(unsigned int x = 0; x < RANGES_SIZE_X; x++)
		for(unsigned int y = 0; y < RANGES_SIZE_Y; y++)
			EXPECT_FLOAT_EQ((result[{x, y}]), x*y);

	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(axes.size(), 2);
	EXPECT_EQ(axes[0].size(), RANGES_SIZE_X);
	EXPECT_EQ(axes[1].size(), RANGES_SIZE_Y);
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_EQ(axes[n][c], c);
}

//TBTKFeature Utilities.PropertyConverter.convert.3 2019-11-26
TEST_F(PropertyConverterTest, convert3){
	AnnotatedArray<double, Subindex> result
		= PropertyConverter::convert(ldosRanges);

	const std::vector<unsigned int> &ranges = result.getRanges();
	EXPECT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], RANGES_SIZE_X);
	EXPECT_EQ(ranges[1], RANGES_SIZE_Y);
	EXPECT_EQ(ranges[2], ENERGY_RESOLUTION);
	for(unsigned int x = 0; x < RANGES_SIZE_X; x++){
		for(unsigned int y = 0; y < RANGES_SIZE_Y; y++){
			for(
				unsigned int e = 0;
				e < (unsigned int)ENERGY_RESOLUTION;
				e++
			){
				EXPECT_FLOAT_EQ((result[{x, y, e}]), x*y*e);
			}
		}
	}

	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(axes.size(), 3);
	EXPECT_EQ(axes[0].size(), RANGES_SIZE_X);
	EXPECT_EQ(axes[1].size(), RANGES_SIZE_Y);
	EXPECT_EQ(axes[2].size(), ENERGY_RESOLUTION);
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_EQ(axes[n][c], c);
}

//TBTKFeature Utilities.PropertyConverter.convert.4 2019-11-26
TEST_F(PropertyConverterTest, convert4){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PropertyConverter::convert(densityCustom);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
