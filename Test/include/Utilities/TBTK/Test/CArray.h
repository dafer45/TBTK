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

//TBTKFeature Utilities.CArray.getSize.1 2019-10-30
TEST(CArray, getSize){
	CArray<unsigned int> carray(10);
	EXPECT_EQ(carray.getSize(), 10);
}

};
