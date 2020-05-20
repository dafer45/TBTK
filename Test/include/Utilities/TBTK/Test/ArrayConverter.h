#include "TBTK/ArrayConverter.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.ArrayConverter.packColumns.0 2020-05-19
TEST(Array, packColumns0){
	Array<unsigned int> array = ArrayConverter::packColumns({
		Vector3d({0, 1, 2}),
		Vector3d({3, 4, 5})
	});
	const std::vector<unsigned int> &ranges = array.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 3);
	EXPECT_EQ(ranges[1], 2);
	for(unsigned int row = 0; row < ranges[0]; row++)
		for(unsigned int column = 0; column < ranges[1]; column++)
			EXPECT_EQ((array[{row, column}]), row + 3*column);
}

//TBTKFeature Utilities.ArrayConverter.packColumns.1 2020-05-19
TEST(Array, packColumns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			ArrayConverter::packColumns({});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.ArrayConverter.packRows.0 2020-05-19
TEST(Array, packRows0){
	Array<unsigned int> array = ArrayConverter::packRows({
		Vector3d({0, 1, 2}),
		Vector3d({3, 4, 5})
	});
	const std::vector<unsigned int> &ranges = array.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	for(unsigned int row = 0; row < ranges[0]; row++)
		for(unsigned int column = 0; column < ranges[1]; column++)
			EXPECT_EQ((array[{row, column}]), 3*row + column);
}

//TBTKFeature Utilities.ArrayConverter.packColumns.1 2020-05-19
TEST(Array, packRows1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			ArrayConverter::packRows({});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.ArrayConverter.splitColumnsToVector3d.0 2020-05-19
TEST(Array, splitColumnsToVector3d0){
	Array<double> array({3, 2});
	for(unsigned int row = 0; row < 3; row++)
		for(unsigned int column = 0; column < 2; column++)
			array[{row, column}] = row + 3*column;

	std::vector<Vector3d> vectors
		= ArrayConverter::splitColumnsToVector3d(array);

	EXPECT_EQ(vectors.size(), 2);
	for(unsigned int column = 0; column < vectors.size(); column++){
		EXPECT_EQ(vectors[column].x, 0 + 3*column);
		EXPECT_EQ(vectors[column].y, 1 + 3*column);
		EXPECT_EQ(vectors[column].z, 2 + 3*column);
	}
}

//TBTKFeature Utilities.ArrayConverter.splitColumnsToVector3d.1 2020-05-19
TEST(Array, splitColumnsToVector3d1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			ArrayConverter::splitColumnsToVector3d(
				Array<double>({4, 2})
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.ArrayConverter.splitRowsToVector3d.0 2020-05-19
TEST(Array, splitRowsToVector3d0){
	Array<double> array({2, 3});
	for(unsigned int row = 0; row < 2; row++)
		for(unsigned int column = 0; column < 3; column++)
			array[{row, column}] = 3*row + column;

	std::vector<Vector3d> vectors
		= ArrayConverter::splitRowsToVector3d(array);

	EXPECT_EQ(vectors.size(), 2);
	for(unsigned int row = 0; row < vectors.size(); row++){
		EXPECT_EQ(vectors[row].x, 0 + 3*row);
		EXPECT_EQ(vectors[row].y, 1 + 3*row);
		EXPECT_EQ(vectors[row].z, 2 + 3*row);
	}
}

//TBTKFeature Utilities.ArrayConverter.splitRowsToVector3d.1 2020-05-19
TEST(Array, splitRowsToVector3d1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			ArrayConverter::splitRowsToVector3d(
				Array<double>({2, 4})
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};
