#include "TBTK/Array.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.Array.construction.1 2019-10-31
TEST(Array, constructor0){
	Array<unsigned int> array;
}

//TBTKFeature Utilities.Array.construction.2 2019-10-31
TEST(Array, constructor1){
	Array<unsigned int> array({2, 3, 4});

	const std::vector<unsigned int> &ranges = array.getRanges();
	EXPECT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);

	EXPECT_EQ(array.getSize(), 2*3*4);
}

//TBTKFeature Utilities.Array.construction.3 2019-10-31
TEST(Array, constructor2){
	Array<unsigned int> array({2, 3, 4}, 5);

	const std::vector<unsigned int> &ranges = array.getRanges();
	EXPECT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			for(unsigned int k = 0; k < 4; k++)
				EXPECT_EQ((array[{i, j, k}]), 5);

	EXPECT_EQ(array.getSize(), 2*3*4);
}

//TBTKFeature Utilities.Array.construction.4 2019-10-31
TEST(Array, constructor3){
	std::vector<unsigned int> vector({2, 3, 4});
	Array<unsigned int> array(vector);

	const std::vector<unsigned int> &ranges = array.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], 3);
	for(unsigned int n = 0; n < ranges[0]; n++)
		EXPECT_EQ(array[{n}], vector[n]);

	EXPECT_EQ(array.getSize(), 3);
}

//TBTKFeature Utilities.Array.construction.5 2019-10-31
TEST(Array, constructor4){
	std::vector<std::vector<unsigned int>> vector(
		2,
		std::vector<unsigned int>(3)
	);
	for(unsigned int x = 0; x < vector.size(); x++)
		for(unsigned int y = 0; y < vector[x].size(); y++)
			vector[x][y] = x*y;
	Array<unsigned int> array(vector);

	const std::vector<unsigned int> &ranges = array.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	for(unsigned int x = 0; x < ranges[0]; x++)
		for(unsigned int y = 0; y < ranges[1]; y++)
			EXPECT_EQ((array[{x, y}]), vector[x][y]);

	EXPECT_EQ(array.getSize(), 2*3);
}

//TBTKFeature Utilities.Array.construction.6 2019-10-31
TEST(Array, constructor5){
	std::vector<std::vector<unsigned int>> vector(2);
	vector[0] = std::vector<unsigned int>(3);
	vector[1] = std::vector<unsigned int>(4);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> array(vector);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.create.1 2019-11-25
TEST(Array, create1){
	Array<unsigned int> array0({2, 3, 4});
	Array<unsigned int> array1 = Array<unsigned int>::create(
		std::vector<unsigned int>({2, 3, 4})
	);

	const std::vector<unsigned int> &ranges0 = array0.getRanges();
	const std::vector<unsigned int> &ranges1 = array1.getRanges();
	EXPECT_EQ(ranges0.size(), ranges1.size());
	for(unsigned int n = 0; n < ranges0.size(); n++)
		EXPECT_EQ(ranges0[n], ranges1[n]);
}

//TBTKFeature Utilities.Array.create.2 2019-11-25
TEST(Array, create2){
	Array<unsigned int> array0({2, 3, 4}, 5);
	Array<unsigned int> array1 = Array<unsigned int>::create(
		std::vector<unsigned int>({2, 3, 4}),
		5
	);

	const std::vector<unsigned int> &ranges0 = array0.getRanges();
	const std::vector<unsigned int> &ranges1 = array1.getRanges();
	EXPECT_EQ(ranges0.size(), ranges1.size());
	for(unsigned int n = 0; n < ranges0.size(); n++)
		EXPECT_EQ(ranges0[n], ranges1[n]);
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			for(unsigned int k = 0; k < 4; k++)
				EXPECT_EQ((array0[{i, j, k}]), (array1[{i, j, k}]));
}

//TBTKFeature Utilities.Array.operatorArraySubscript.1 2019-10-31
TEST(Array, operatorArraySubscript0){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((array[{i, j}]), i + 2*j);
}

//TBTKFeature Utilities.Array.operatorArraySubscript.2.C++ 2019-10-31
TEST(Array, operatorArraySubscript1){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	const Array<unsigned int> &constArray = array;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((constArray[{i, j}]), i + 2*j);
}

//TBTKFeature Utilities.Array.operatorArraySubscript.3 2019-10-31
TEST(Array, operatorArraySubscript2){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ(array[3*i + j], i + 2*j);
}

//TBTKFeature Utilities.Array.operatorArraySubscript.4.C++ 2019-10-31
TEST(Array, operatorArraySubscript3){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	const Array<unsigned int> constArray = array;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ(constArray[3*i + j], i + 2*j);
}

//TBTKFeature Utilities.Array.operatorAddition.1 2019-10-31
TEST(Array, operatorAddition0){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 3});
	for(unsigned int i = 0; i < 2; i++){
		for(unsigned int j = 0; j < 3; j++){
			array0[{i, j}] = i;
			array1[{i, j}] = 2*j;
		}
	}

	Array<unsigned int> sum = array0 + array1;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((sum[{i, j}]), i + 2*j);
}

//TBTKFeature Utilities.Array.operatorAddition.2 2019-10-31
TEST(Array, operatorAddition1){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 3, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 + array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 + array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorAddition.3 2019-10-31
TEST(Array, operatorAddition2){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 + array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 + array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorSubtraction.1 2019-10-31
TEST(Array, operatorSubtraction0){
	Array<int> array0({2, 3});
	Array<int> array1({2, 3});
	for(unsigned int i = 0; i < 2; i++){
		for(unsigned int j = 0; j < 3; j++){
			array0[{i, j}] = i;
			array1[{i, j}] = 2*j;
		}
	}

	Array<int> sum = array0 - array1;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((sum[{i, j}]), (int)i - (int)2*j);
}

//TBTKFeature Utilities.Array.operatorSubtraction.2 2019-10-31
TEST(Array, operatorSubtraction1){
	Array<int> array0({2, 3});
	Array<int> array1({2, 3, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 - array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 - array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorSubtraction.3 2019-10-31
TEST(Array, operatorSubtraction2){
	Array<int> array0({2, 3});
	Array<int> array1({2, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 - array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 - array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorMultiplication.1 2019-10-31
TEST(Array, operatorMultiplication0){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> product = array*3;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((product[{i, j}]), 3*(i + 2*j));
}

//TBTKFeature Utilities.Array.operatorMultiplication.2 2019-10-31
TEST(Array, operatorMultiplication1){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> product = 3*array;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((product[{i, j}]), 3*(i + 2*j));
}

//TBTKFeature Utilities.Array.operatorDivision.1 2019-10-31
TEST(Array, operatorDivision0){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> product = array/3;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((product[{i, j}]), (i + 2*j)/3);
}

//TBTKFeature Utilities.Array.operatorComparison.1 2019-11-26
TEST(Array, operatorComparison0){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 3});
	for(unsigned int x = 0; x < 2; x++){
		for(unsigned int y = 0; y < 3; y++){
			array0[{x, y}] = x*y;
			array1[{x, y}] = x*y;
		}
	}

	EXPECT_EQ(array0, array1);
}

//TBTKFeature Utilities.Array.operatorComparison.2 2019-11-26
TEST(Array, operatorComparison1){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2});

	EXPECT_FALSE(array0 == array1);
}

//TBTKFeature Utilities.Array.operatorComparison.3 2019-11-26
TEST(Array, operatorComparison2){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 2});

	EXPECT_FALSE(array0 == array1);
}

//TBTKFeature Utilities.Array.operatorComparison.4 2019-11-26
TEST(Array, operatorComparison3){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 3});
	for(unsigned int x = 0; x < 2; x++){
		for(unsigned int y = 0; y < 3; y++){
			array0[{x, y}] = x*y;
			array1[{x, y}] = x*y;
		}
	}
	array1[{1,1}] = 100;

	EXPECT_FALSE(array0 == array1);
}

//TBTKFeature Utilities.Array.getSlice.1 2019-10-31
TEST(Array, getSlice){
	Array<unsigned int> array({2, 3, 4});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			for(unsigned int k = 0; k < 4; k++)
				array[{i, j, k}] = i + 2*j;

	Array<unsigned int> slicedArray = array.getSlice({_a_, 2, _a_});

	const std::vector<unsigned int> &ranges = slicedArray.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 4);
	EXPECT_EQ(slicedArray.getSize(), 2*4);

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int k = 0; k < 4; k++)
			EXPECT_EQ((slicedArray[{i, k}]), (array[{i, 2, k}]));
}

//TBTKFeature Utilities.Array.getPermutation.1 2019-12-22
TEST(Array, getPermutation1){
	Array<int> array({2, 3, 4, 5});
	int counter = 0;
	for(unsigned int k = 0; k < 2; k++)
		for(unsigned int l = 0; l < 3; l++)
			for(unsigned int m = 0; m < 4; m++)
				for(unsigned int n = 0; n < 5; n++)
					array[{k, l, m, n}] = counter++;

	Array<int> permutation = array.getPermutation({2, 3, 1, 0});
	const std::vector<unsigned int> &ranges = permutation.getRanges();
	EXPECT_EQ(ranges.size(), 4);
	EXPECT_EQ(ranges[0], 4);
	EXPECT_EQ(ranges[1], 5);
	EXPECT_EQ(ranges[2], 3);
	EXPECT_EQ(ranges[3], 2);

	counter = 0;
	for(unsigned int k = 0; k < 2; k++){
		for(unsigned int l = 0; l < 3; l++){
			for(unsigned int m = 0; m < 4; m++){
				for(unsigned int n = 0; n < 5; n++){
					EXPECT_EQ(
						(permutation[{m, n, l, k}]),
						counter++
					);
				}
			}
		}
	}
}

//TBTKFeature Utilities.Array.getPermutation.2 2019-12-22
TEST(Array, getPermutation2){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getPermutation({0, 1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getPermutation.3 2019-12-22
TEST(Array, getPermutation3){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getPermutation({0, 1, 2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getPermutation.4 2019-12-22
TEST(Array, getPermutation4){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getPermutation({0, 1, 4, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getPermutation.5 2019-12-22
TEST(Array, getPermutation5){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getPermutation({0, 1, 1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getData.1 2019-10-31
TEST(Array, getData0){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	CArray<unsigned int> &rawData = array.getData();
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ(rawData[3*i + j], (array[{i, j}]));
}

//TBTKFeature Utilities.Array.getData.2.C++ 2019-10-31
TEST(Array, getData1){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	const CArray<unsigned int> &rawData
		= ((const Array<unsigned int>&)array).getData();
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ(rawData[3*i + j], (array[{i, j}]));
}

//TBTKFeature Utilities.Array.getSize.1 2019-10-31
TEST(Array, getSize){
	Array<unsigned int> array({2, 3, 4});
	EXPECT_EQ(array.getSize(), 2*3*4);
}

};
