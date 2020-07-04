#include "TBTK/CArray.h"

#include "gtest/gtest.h"

namespace TBTK{

//TBTKFeature Utilities.CArray.construction.1 2019-10-30
TEST(CArray, constructor0){
	CArray<unsigned int> carray;
}

//TBTKFeature Utilities.CArray.construction.2 2019-10-30
TEST(CArray, constructor1){
	CArray<unsigned int> carray(10);
}

//TBTKFeature Utilities.CArray.construction.3 2020-07-04
TEST(CArray, constructor2){
	CArray<unsigned int> carray = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(carray[n], n);
}

//TBTKFeature Utilities.CArray.copyConstruction.1.C++ 2019-10-30
TEST(CArray, copyConstructor0){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	CArray<unsigned int> copyCArray = carray;
	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(copyCArray[n], carray[n]);
	EXPECT_EQ(copyCArray.getSize(), carray.getSize());
}

//TBTKFeature Utilities.CArray.moveConstruction.1.C++ 2019-10-30
TEST(CArray, moveConstructor0){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	CArray<unsigned int> movedCArray = std::move(carray);
	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(movedCArray[n], n);
	EXPECT_EQ(movedCArray.getSize(), 10);
}

//TBTKFeature Utilities.CArray.serializeToJSON.1 2020-02-03
TEST(CArray, serializeToJSON0){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	CArray<unsigned int> copy(
		carray.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(copy.getSize(), carray.getSize());
	for(unsigned int n = 0; n < copy.getSize(); n++)
		EXPECT_EQ(copy[n], carray[n]);
}

//TBTKFeature Utilities.CArray.asignmentOperator.1.C++ 2019-10-30
TEST(CArray, assignmentOperator){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	CArray<unsigned int> copyCArray;
	copyCArray = carray;
	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(copyCArray[n], carray[n]);
	EXPECT_EQ(copyCArray.getSize(), carray.getSize());
}

//TBTKFeature Utilities.CArray.copyConstruction.1.C++ 2019-10-30
TEST(CArray, moveAssignmentOperator){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	CArray<unsigned int> movedCArray;
	movedCArray = std::move(carray);
	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(movedCArray[n], n);
	EXPECT_EQ(movedCArray.getSize(), 10);
}

//TBTKFeature Utilities.CArray.operatorArraySubscript.1.C++ 2019-10-30
TEST(CArray, operatorArraySubscript0){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(carray[n], n);
}

//TBTKFeature Utilities.CArray.operatorArraySubscript.2.C++ 2019-10-30
TEST(CArray, operatorArraySubscript1){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	const CArray<unsigned int> &constCArray = carray;
	for(unsigned int n = 0; n < 10; n++)
		EXPECT_EQ(constCArray[n], n);
}

//TBTKFeature Utilities.CArray.getData.1.C++ 2019-10-31
TEST(CArray, getData0){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	unsigned int *data = carray.getData();
	for(unsigned int n = 0; n < carray.getSize(); n++)
		EXPECT_EQ(data[n], n);
}

//TBTKFeature Utilities.CArray.getData.2.C++ 2019-10-31
TEST(CArray, getData1){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < 10; n++)
		carray[n] = n;

	const unsigned int *data
		= ((const CArray<unsigned int>&)carray).getData();
	for(unsigned int n = 0; n < carray.getSize(); n++)
		EXPECT_EQ(data[n], n);
}

//TBTKFeature Utilities.CArray.getSize.1 2019-10-30
TEST(CArray, getSize){
	CArray<unsigned int> carray(10);
	EXPECT_EQ(carray.getSize(), 10);
}

};
