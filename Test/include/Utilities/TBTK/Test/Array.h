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

//TBTKFeature Utilities.Array.copy.1 2019-10-31
TEST(Array, copyConstructor){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> copyArray = array;

	const std::vector<unsigned int> &ranges = copyArray.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 2; j++)
			EXPECT_EQ((copyArray[{i, j}]), (array[{i, j}]));

	EXPECT_EQ(copyArray.getSize(), array.getSize());
}

//TBTKFeature Utilities.Array.move.1 2019-10-31
TEST(Array, moveConstructor){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> movedArray = std::move(array);

	const std::vector<unsigned int> &ranges = movedArray.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 2; j++)
			EXPECT_EQ((movedArray[{i, j}]), i + 2*j);

	EXPECT_EQ(movedArray.getSize(), 2*3);
}

//TBTKFeature Utilities.Array.copyAssignment.1 2019-10-31
TEST(Array, operatorAssignment){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> copyArray;
	copyArray = array;

	const std::vector<unsigned int> &ranges = copyArray.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 2; j++)
			EXPECT_EQ((copyArray[{i, j}]), (array[{i, j}]));

	EXPECT_EQ(copyArray.getSize(), array.getSize());
}

//TBTKFeature Utilities.Array.moveAssignment.1 2019-10-31
TEST(Array, operatorMoveAssignment){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	Array<unsigned int> movedArray;
	movedArray = std::move(array);

	const std::vector<unsigned int> &ranges = movedArray.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);

	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 2; j++)
			EXPECT_EQ((movedArray[{i, j}]), i + 2*j);

	EXPECT_EQ(movedArray.getSize(), 2*3);
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

//TBTKFeature Utilities.Array.getSize.1 2019-10-31
TEST(Array, getSize){
	Array<unsigned int> array({2, 3, 4});
	EXPECT_EQ(array.getSize(), 2*3*4);
}

};
