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

//TBTKFeature Utilities.Array.construction.7 2020-05-19
TEST(Array, construction6){
	for(unsigned int n = 0; n < 4; n++){
		Range range(1, 2, 10, (bool)(n%2), (bool)(n/2));
		Array<double> array = range;
		const std::vector<unsigned int> &ranges = array.getRanges();
		EXPECT_EQ(ranges.size(), 1);
		EXPECT_EQ(ranges[0], 10);
		for(unsigned int c = 0; c < ranges[0]; c++)
			EXPECT_EQ(array[{c}], range[c]);
	}
}

//TBTKFeature Utilities.Array.serializeToJSON.1 2020-02-03
TEST(Array, serializeToJSON1){
	Array<unsigned int> array({2, 4});
	for(unsigned int n = 0; n < 2; n++)
		for(unsigned int c = 0; c < 4; c++)
			array[{n, c}] = n*c;

	Array<unsigned int> copy(
		array.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	const std::vector<unsigned int> &ranges = copy.getRanges();
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 4);
	for(unsigned int n = 0; n < 2; n++)
		for(unsigned int c = 0; c < 4; c++)
			EXPECT_EQ((copy[{n, c}]), (array[{n, c}]));
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

//TBTKFeature Utilities.Array.operatorTypeCast.1 2020-05-18
TEST(Array, operatorTypeCast1){
	Array<unsigned int> array0({2, 3, 4});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			for(unsigned int k = 0; k < 4; k++)
				array0[{i, j, k}] = i*j*k;

	Array<int> array1 = array0;
	const std::vector<unsigned int> &ranges = array1.getRanges();
	EXPECT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	EXPECT_EQ(ranges[2], 4);
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			for(unsigned int k = 0; k < 4; k++)
				EXPECT_EQ((array1[{i, j, k}]), (int)(i*j*k));
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

//TBTKFeature Utilities.Array.operatorAdditionEquality.1 2020-05-25
TEST(Array, operatorAdditionEquality1){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 3});
	for(unsigned int i = 0; i < 2; i++){
		for(unsigned int j = 0; j < 3; j++){
			array0[{i, j}] = i;
			array1[{i, j}] = 2*j;
		}
	}

	array0 += array1;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((array0[{i, j}]), i + 2*j);
}

//TBTKFeature Utilities.Array.operatorAdditionEquality.2 2020-05-25
TEST(Array, operatorAdditionEquality2){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 3, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 += array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 += array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorAdditionEquality.3 2020-05-25
TEST(Array, operatorAddition3){
	Array<unsigned int> array0({2, 3});
	Array<unsigned int> array1({2, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 += array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 += array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
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

//TBTKFeature Utilities.Array.operatorSubtractionEquality.1 2020-05-25
TEST(Array, operatorSubtractionEquality0){
	Array<int> array0({2, 3});
	Array<int> array1({2, 3});
	for(unsigned int i = 0; i < 2; i++){
		for(unsigned int j = 0; j < 3; j++){
			array0[{i, j}] = i;
			array1[{i, j}] = 2*j;
		}
	}

	array0 -= array1;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((array0[{i, j}]), (int)i - (int)2*j);
}

//TBTKFeature Utilities.Array.operatorSubtractionEquality.2 2020-05-25
TEST(Array, operatorSubtractionEquality1){
	Array<int> array0({2, 3});
	Array<int> array1({2, 3, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 -= array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 -= array0;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorSubtractionEquality.3 2020-05-25
TEST(Array, operatorSubtractionEqiality2){
	Array<int> array0({2, 3});
	Array<int> array1({2, 4});

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array0 -= array1;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array1 -= array0;
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

//TBTKFeature Utilities.Array.operatorNegative.1 2020-05-24
TEST(Array, operatorNegative){
	Array<int> array({2, 3});
	for(int n = 0; n < 2; n++)
		for(int c = 0; c < 3; c++)
			array[{(unsigned int)n, (unsigned int)c}] = 3*n + c;

	Array<int> result = -array;
	const std::vector<unsigned int> &ranges = result.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 3);
	for(int n = 0; n < 2; n++){
		for(int c = 0; c < 3; c++){
			EXPECT_EQ(
				(result[{(unsigned int)n, (unsigned int)c}]),
				-(3*n + c)
			);
		}
	}
}

//TBTKFeature Utilities.Array.operatorMultiplicationEquality.1 2020-05-25
TEST(Array, operatorMultiplicationEquality0){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	array *= 3;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((array[{i, j}]), 3*(i + 2*j));
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

//TBTKFeature Utilities.Array.operatorMultiplication.3 2020-05-17
TEST(Array, operatorMultiplication3){
	Array<unsigned int> u({3});
	Array<unsigned int> v({3});
	for(unsigned int n = 0; n < 3; n++){
		u[{n}] = n;
		v[{n}] = 2*n;
	}
	Array<unsigned int> s = u*v;

	const std::vector<unsigned int> &ranges = s.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], 1);
	EXPECT_EQ(s[{0}], 10);
}

//TBTKFeature Utilities.Array.operatorMultiplication.4 2020-05-17
TEST(Array, operatorMultiplication4){
	Array<unsigned int> u({3});
	Array<unsigned int> v({4});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			u*v;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorMultiplication.5 2020-05-17
TEST(Array, operatorMultiplication5){
	Array<unsigned int> u({3});
	Array<unsigned int> M({3, 2});
	for(unsigned int n = 0; n < 3; n++){
		u[{n}] = n;
		for(unsigned int c = 0; c < 2; c++)
			M[{n, c}] = n*(c+1);
	}
	Array<unsigned int> w = u*M;

	const std::vector<unsigned int> &ranges = w.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(w[{0}], 5);
	EXPECT_EQ(w[{1}], 10);
}

//TBTKFeature Utilities.Array.operatorMultiplication.6 2020-05-17
TEST(Array, operatorMultiplication6){
	Array<unsigned int> u({3});
	Array<unsigned int> M({4, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			u*M;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorMultiplication.7 2020-05-17
TEST(Array, operatorMultiplication7){
	Array<unsigned int> u({3});
	Array<unsigned int> M({2, 3});
	for(unsigned int n = 0; n < 3; n++){
		u[{n}] = n;
		for(unsigned int c = 0; c < 2; c++)
			M[{c, n}] = n*(c+1);
	}
	Array<unsigned int> w = M*u;

	const std::vector<unsigned int> &ranges = w.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(w[{0}], 5);
	EXPECT_EQ(w[{1}], 10);
}

//TBTKFeature Utilities.Array.operatorMultiplication.8 2020-05-17
TEST(Array, operatorMultiplication8){
	Array<unsigned int> u({3});
	Array<unsigned int> M({3, 4});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			M*u;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorMultiplication.9 2020-05-17
TEST(Array, operatorMultiplication9){
	Array<unsigned int> M({2, 3});
	Array<unsigned int> N({3, 4});
	for(unsigned int n = 0; n < 2; n++)
		for(unsigned int c = 0; c < 3; c++)
			M[{n, c}] = (n+1)*c;
	for(unsigned int n = 0; n < 3; n++)
		for(unsigned int c = 0; c < 4; c++)
			N[{n, c}] = (n+1)*(c+1);

	Array<unsigned int> A = M*N;

	const std::vector<unsigned int> &ranges = A.getRanges();
	EXPECT_EQ(ranges.size(), 2);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 4);
	for(unsigned int n = 0; n < 2; n++)
		for(unsigned int c = 0; c < 4; c++)
			EXPECT_EQ((A[{n, c}]), 8*(n+1)*(c+1));
}

//TBTKFeature Utilities.Array.operatorMultiplication.10 2020-05-17
TEST(Array, operatorMultiplication10){
	Array<unsigned int> M({2, 3});
	Array<unsigned int> N({4, 4});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			M*N;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorMultiplication.11 2020-05-17
TEST(Array, operatorMultiplication11){
	Array<unsigned int> u({2});
	Array<unsigned int> N({2, 3, 4});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			u*N;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.operatorMultiplication.12 2020-05-17
TEST(Array, operatorMultiplication12){
	Array<unsigned int> u({4});
	Array<unsigned int> N({2, 3, 4});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			N*u;
		},
		::testing::ExitedWithCode(1),
		""
	);
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

//TBTKFeature Utilities.Array.operatorDivisionEquality.1 2020-05-25
TEST(Array, operatorDivisionEquality0){
	Array<unsigned int> array({2, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			array[{i, j}] = i + 2*j;

	array /= 3;
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			EXPECT_EQ((array[{i, j}]), (i + 2*j)/3);
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

//TBTKFeature Utilities.Array.contract.1 2020-05-17
TEST(Array, contract1){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	for(unsigned int i = 0; i < 2; i++)
		for(unsigned int j = 0; j < 3; j++)
			for(unsigned int k = 0; k < 2; k++)
				for(unsigned int l = 0; l < 2; l++)
					B[{i, j, k, l}] = i*j*k*l;
	for(unsigned int i = 0; i < 4; i++)
		for(unsigned int j = 0; j < 6; j++)
			for(unsigned int k = 0; k < 5; k++)
				C[{i, j, k}] = i*j*k;

	Array<unsigned int> A = Array<unsigned int>::contract(
		B,
		{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL},
		C,
		{IDX_ALL_(0), IDX_ALL, IDX_ALL_(1)}
	);

	const std::vector<unsigned int> &ranges = A.getRanges();
	EXPECT_EQ(ranges.size(), 3);
	EXPECT_EQ(ranges[0], 2);
	EXPECT_EQ(ranges[1], 5);
	EXPECT_EQ(ranges[2], 6);
	for(unsigned int i = 0; i < 2; i++){
		for(unsigned int j = 0; j < 5; j++){
			for(unsigned int k = 0; k < 6; k++){
				unsigned int result = 0;
				for(unsigned int m = 0; m < 4; m++){
					for(unsigned int n = 0; n < 3; n++){
						result += B[{i, n, m, j}]*C[
							{m, k, n}
						];
					}
				}
				EXPECT_EQ((A[{i, j, k}]), result);
			}
		}
	}
}

//TBTKFeature Utilities.Array.contract.2 2020-05-17
TEST(Array, contract2){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(0), IDX_ALL_(1), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(1)}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.3 2020-05-17
TEST(Array, contract3){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL, IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(1)}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.4 2020-05-17
TEST(Array, contract4){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.5 2020-05-17
TEST(Array, contract5){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(2)}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.6 2020-05-17
TEST(Array, contract6){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL, IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(2)}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.7 2020-05-17
TEST(Array, contract7){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(2), IDX_ALL}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.8 2020-05-17
TEST(Array, contract8){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), 0},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(2), IDX_ALL}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.9 2020-05-17
TEST(Array, contract9){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(2), 0}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.10 2020-05-17
TEST(Array, contract10){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(0), IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(1), 0}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.11 2020-05-17
TEST(Array, contract11){
	Array<unsigned int> B({2, 3, 4, 5});
	Array<unsigned int> C({4, 6, 3});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Array<unsigned int> A = Array<unsigned int>::contract(
				B,
				{IDX_ALL, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL},
				C,
				{IDX_ALL_(0), IDX_ALL, IDX_ALL_(0), 0}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.contract.12 2020-05-17
TEST(Array, contract12){
	Array<unsigned int> B({2, 3});
	Array<unsigned int> C({3, 2});
	double contraction = 0;
	for(unsigned int i = 0; i < 2; i++){
		for(unsigned int j = 0; j < 3; j++){
			B[{i, j}] = i+j;
			C[{j, i}] = i+2*j;
			contraction += (i+j)*(i+2*j);
		}
	}
	Array<unsigned int> A = Array<unsigned int>::contract(
		B,
		{IDX_ALL_(0), IDX_ALL_(1)},
		C,
		{IDX_ALL_(1), IDX_ALL_(0)}
	);
	const std::vector<unsigned int> &ranges = A.getRanges();
	EXPECT_EQ(ranges.size(), 1);
	EXPECT_EQ(ranges[0], 1);
	EXPECT_EQ(A[{0}], contraction);
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

//TBTKFeature Utilities.Array.getArrayWithPermutedIndices.1 2019-12-22
TEST(Array, getArrayWithPermutedIndices1){
	Array<int> array({2, 3, 4, 5});
	int counter = 0;
	for(unsigned int k = 0; k < 2; k++)
		for(unsigned int l = 0; l < 3; l++)
			for(unsigned int m = 0; m < 4; m++)
				for(unsigned int n = 0; n < 5; n++)
					array[{k, l, m, n}] = counter++;

	Array<int> permutation
		= array.getArrayWithPermutedIndices({2, 3, 1, 0});
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

//TBTKFeature Utilities.Array.getArrayWithPermutedIndices.2 2019-12-22
TEST(Array, getArrayWithPermutedIndices2){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getArrayWithPermutedIndices({0, 1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getArrayWithPermutedIndices.3 2019-12-22
TEST(Array, getArrayWithPermutedIndices3){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getArrayWithPermutedIndices({0, 1, 2, 3, 4});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getArrayWithPermutedIndices.4 2019-12-22
TEST(Array, getArrayWithPermutedIndices4){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getArrayWithPermutedIndices({0, 1, 4, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getArrayWithPermutedIndices.5 2019-12-22
TEST(Array, getArrayWithPermutedIndices5){
	Array<int> array({2, 3, 4, 5});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			array.getArrayWithPermutedIndices({0, 1, 1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Utilities.Array.getArrayWithReversedIndices.1 2019-12-22
TEST(Array, getArrayWithReveredIndices1){
	Array<int> array({2, 3, 4, 5});
	int counter = 0;
	for(unsigned int k = 0; k < 2; k++)
		for(unsigned int l = 0; l < 3; l++)
			for(unsigned int m = 0; m < 4; m++)
				for(unsigned int n = 0; n < 5; n++)
					array[{k, l, m, n}] = counter++;

	Array<int> result = array.getArrayWithReversedIndices();
	const std::vector<unsigned int> &ranges = result.getRanges();
	EXPECT_EQ(ranges[0], 5);
	EXPECT_EQ(ranges[1], 4);
	EXPECT_EQ(ranges[2], 3);
	EXPECT_EQ(ranges[3], 2);
	for(unsigned int k = 0; k < 2; k++){
		for(unsigned int l = 0; l < 3; l++){
			for(unsigned int m = 0; m < 4; m++){
				for(unsigned int n = 0; n < 5; n++){
					EXPECT_EQ(
						(result[{n, m, l, k}]),
						(array[{k, l, m, n}])
					);
				}
			}
		}
	}
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
