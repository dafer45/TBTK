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
	Property::LDOS ldosCustom;

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
		SetUpLDOSCustom();
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

	void SetUpLDOSCustom(){
		IndexTree indexTree;
		indexTree.add({1, 4});
		indexTree.add({2, 2});
		indexTree.add({2, 3});
		indexTree.generateLinearMap();
		ldosCustom = Property::LDOS(
			indexTree,
			ENERGY_LOWER_BOUND,
			ENERGY_UPPER_BOUND,
			ENERGY_RESOLUTION
		);

		for(
			unsigned int n = 0;
			n < (unsigned int)ENERGY_RESOLUTION;
			n++
		){
			ldosCustom({1, 4}, n) = 1*n;
			ldosCustom({2, 2}, n) = 2*n;
			ldosCustom({2, 3}, n) = 3*n;
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

//TBTKFeature Utilities.PropertyConverter.convert.5 2019-11-26
TEST_F(PropertyConverterTest, convert5){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PropertyConverter::convert(dos, {});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.PropertyConverter.convert.6 2019-11-26
TEST_F(PropertyConverterTest, convert6){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PropertyConverter::convert(
				densityRanges,
				{IDX_ALL, IDX_ALL}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.PropertyConverter.convert.7 2019-11-26
TEST_F(PropertyConverterTest, convert7){
	AnnotatedArray<double, Subindex> result = PropertyConverter::convert(
		densityCustom,
		{IDX_ALL, IDX_ALL}
	);

	//The Indices are {1, 4}, {2, 2}, {2, 3}.
	const Subindex LOWER[2] = {1, 2};
	const Subindex UPPER[2] = {2, 4};
	const Subindex SIZE[2] = {
		UPPER[0] - LOWER[0] + 1,
		UPPER[1] - LOWER[1] + 1
	};

	const std::vector<unsigned int> &ranges = result.getRanges();
	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], SIZE[0]);
	EXPECT_EQ(ranges[1], SIZE[1]);
	for(unsigned int x = 0; x < ranges[0]; x++){
		for(unsigned int y = 0; y < ranges[1]; y++){
			Subindex X = axes[0][x];
			Subindex Y = axes[1][y];
			if(X == 1 && Y == 4)
				EXPECT_FLOAT_EQ((result[{x, y}]), 1);
			else if(X == 2 && Y == 2)
				EXPECT_FLOAT_EQ((result[{x, y}]), 2);
			else if(X == 2 && Y == 3)
				EXPECT_FLOAT_EQ((result[{x, y}]), 3);
			else
				EXPECT_FLOAT_EQ((result[{x, y}]), 0);
		}
	}

	EXPECT_EQ(axes.size(), 2);
	EXPECT_EQ(axes[0].size(), SIZE[0]);
	EXPECT_EQ(axes[1].size(), SIZE[1]);
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_EQ(axes[n][c], LOWER[n] + c);
}

//TBTKFeature Utilities.PropertyConverter.convert.8 2019-11-26
TEST_F(PropertyConverterTest, convert8){
	AnnotatedArray<double, Subindex> result = PropertyConverter::convert(
		densityCustom,
		{2, IDX_ALL}
	);

	//The compatible Indices {2, 2}, {2, 3}.
	const Subindex LOWER = 2;
	const Subindex UPPER = 3;
	const Subindex SIZE = UPPER - LOWER + 1;

	const std::vector<unsigned int> &ranges = result.getRanges();
	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], SIZE);
	for(unsigned int x = 0; x < ranges[0]; x++){
		for(unsigned int y = 0; y < ranges[1]; y++){
			Subindex X = axes[0][x];
			Subindex Y = axes[1][y];
			if(X == 2 && Y == 2)
				EXPECT_FLOAT_EQ((result[{x, y}]), 2);
			else if(X == 2 && Y == 3)
				EXPECT_FLOAT_EQ((result[{x, y}]), 3);
			else
				EXPECT_FLOAT_EQ((result[{x, y}]), 0);
		}
	}

	EXPECT_EQ(axes.size(), 1);
	EXPECT_EQ(axes[0].size(), SIZE);
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_EQ(axes[n][c], LOWER + c);
}

//TBTKFeature Utilities.PropertyConverter.convert.9 2019-11-26
TEST_F(PropertyConverterTest, convert9){
	AnnotatedArray<double, Subindex> result
		= PropertyConverter::convert(ldosCustom, {IDX_ALL, IDX_ALL});

	//The Indices are {1, 4}, {2, 2}, {2, 3}.
	Subindex LOWER[3] = {1, 2, 0};
	Subindex UPPER[3] = {2, 4, ENERGY_RESOLUTION-1};
	Subindex SIZE[3] = {
		UPPER[0] - LOWER[0] + 1,
		UPPER[1] - LOWER[1] + 1,
		ENERGY_RESOLUTION
	};

	const std::vector<unsigned int> &ranges = result.getRanges();
	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(ranges.size(), 3);
	for(unsigned int n = 0; n < 3; n++)
		EXPECT_EQ(ranges[n], SIZE[n]);
	for(unsigned int x = 0; x < ranges[0]; x++){
		for(unsigned int y = 0; y < ranges[1]; y++){
			Subindex X = axes[0][x];
			Subindex Y = axes[1][y];
			for(
				unsigned int e = 0;
				e < (unsigned int)ENERGY_RESOLUTION;
				e++
			){
				if(X == 1 && Y == 4)
					EXPECT_FLOAT_EQ((result[{x, y, e}]), 1*e);
				else if(X == 2 && Y == 2)
					EXPECT_FLOAT_EQ((result[{x, y, e}]), 2*e);
				else if(X == 2 && Y == 3)
					EXPECT_FLOAT_EQ((result[{x, y, e}]), 3*e);
				else
					EXPECT_FLOAT_EQ((result[{x, y, e}]), 0);
			}
		}
	}

	EXPECT_EQ(axes.size(), 3);
	for(unsigned int n = 0; n < 3; n++)
		EXPECT_EQ(axes[n].size(), SIZE[n]);
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_EQ(axes[n][c], LOWER[n] + c);
}

//TBTKFeature Utilities.PropertyConverter.convert.10 2019-11-26
TEST_F(PropertyConverterTest, convert10){
	AnnotatedArray<double, Subindex> result
		= PropertyConverter::convert(ldosCustom, {IDX_ALL, 4});

	//The compatible Index is {1, 4}.
	Subindex LOWER[2] = {1, 0};
	Subindex UPPER[2] = {1, ENERGY_RESOLUTION-1};
	Subindex SIZE[2] = {
		UPPER[0] - LOWER[0] + 1,
		ENERGY_RESOLUTION
	};

	const std::vector<unsigned int> &ranges = result.getRanges();
	const std::vector<std::vector<Subindex>> &axes = result.getAxes();
	EXPECT_EQ(ranges.size(), 2);
	for(unsigned int n = 0; n < 2; n++)
		EXPECT_EQ(ranges[n], SIZE[n]);
	for(unsigned int x = 0; x < ranges[0]; x++){
		Subindex X = axes[0][x];
		for(
			unsigned int e = 0;
			e < (unsigned int)ENERGY_RESOLUTION;
			e++
		){
			if(X == 1)
				EXPECT_FLOAT_EQ((result[{x, e}]), 1*e);
			else
				EXPECT_FLOAT_EQ((result[{x, e}]), 0);
		}
	}

	EXPECT_EQ(axes.size(), 2);
	for(unsigned int n = 0; n < 2; n++)
		EXPECT_EQ(axes[n].size(), SIZE[n]);
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			EXPECT_EQ(axes[n][c], LOWER[n] + c);
}

};
