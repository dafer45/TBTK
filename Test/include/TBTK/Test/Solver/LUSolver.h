#include "TBTK/Solver/LUSolver.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

TEST(LUSolver, Constructor){
	//Not testable on its own.
}

TEST(LUSolver, Destructor){
	//Not testable on its own.
}

TEST(LUSolver, setMatrix){
	LUSolver solver;

	//Fail to set matrix with no matrix elements.
	SparseMatrix<double> sparseMatrix0(
		SparseMatrix<double>::StorageFormat::CSC
	);
	sparseMatrix0.constructCSX();
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			solver.setMatrix(sparseMatrix0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	::testing::FLAGS_gtest_death_test_style = "fast";

	//Set SparseMatrix<double>
	SparseMatrix<double> sparseMatrix1(
		SparseMatrix<double>::StorageFormat::CSC
	);
	sparseMatrix1.add(0, 0, 1);
	sparseMatrix1.add(0, 1, 0.1);
	sparseMatrix1.add(1, 0, 0.1);
	sparseMatrix1.add(1, 1, 1);
	sparseMatrix1.constructCSX();
	solver.setMatrix(sparseMatrix1);

	//Overwrite matrix.
	SparseMatrix<double> sparseMatrix2(
		SparseMatrix<double>::StorageFormat::CSC
	);
	sparseMatrix2.add(0, 0, 1);
	sparseMatrix2.constructCSX();
	solver.setMatrix(sparseMatrix2);

	//Set SparseMatrix<std::complex<double>>.
	SparseMatrix<std::complex<double>> sparseMatrix3(
		SparseMatrix<std::complex<double>>::StorageFormat::CSC
	);
	sparseMatrix3.add(0, 0, 1);
	sparseMatrix3.add(0, 1, 0.1);
	sparseMatrix3.add(1, 0, 0.1);
	sparseMatrix3.add(1, 1, 1);
	sparseMatrix3.constructCSX();
	solver.setMatrix(sparseMatrix3);
}

TEST(LUSolver, getMatrixDataType){
	LUSolver solver;

	//Initialized with DataType::None.
	EXPECT_EQ(solver.getMatrixDataType(), LUSolver::DataType::None);

	//Set SparseMatrix<double>.
	SparseMatrix<double> sparseMatrix0(
		SparseMatrix<double>::StorageFormat::CSC
	);
	sparseMatrix0.add(0, 0, 1);
	sparseMatrix0.constructCSX();
	solver.setMatrix(sparseMatrix0);
	EXPECT_EQ(solver.getMatrixDataType(), LUSolver::DataType::Double);

	//Set SparseMatrix<std::complex<double>>.
	SparseMatrix<std::complex<double>> sparseMatrix1(
		SparseMatrix<std::complex<double>>::StorageFormat::CSC
	);
	sparseMatrix1.add(0, 0, std::complex<double>(1, 1));
	sparseMatrix1.constructCSX();
	solver.setMatrix(sparseMatrix1);
	EXPECT_EQ(
		solver.getMatrixDataType(),
		LUSolver::DataType::ComplexDouble
	);

	//Recognize real matrix in spite of being passed
	//SparseMatrix<std::complex<double>>.
	SparseMatrix<std::complex<double>> sparseMatrix2(
		SparseMatrix<std::complex<double>>::StorageFormat::CSC
	);
	sparseMatrix2.add(0, 0, 1);
	sparseMatrix2.constructCSX();
	solver.setMatrix(sparseMatrix2);
	EXPECT_EQ(solver.getMatrixDataType(), LUSolver::DataType::Double);
}

//TODO
//...
TEST(LUSolver, solve){
	LUSolver solver;

	//Real matrix.
	SparseMatrix<double> sparseMatrix0(
		SparseMatrix<double>::StorageFormat::CSC
	);
	sparseMatrix0.add(0, 0, 1);
	sparseMatrix0.add(0, 1, 2);
	sparseMatrix0.add(1, 0, 3);
	sparseMatrix0.add(1, 1, 4);
	sparseMatrix0.constructCSX();

	//Complex matrix.
	SparseMatrix<std::complex<double>> sparseMatrix1(
		SparseMatrix<std::complex<double>>::StorageFormat::CSC
	);
	sparseMatrix1.add(0, 0, 1);
	sparseMatrix1.add(0, 1, std::complex<double>(0, 1));
	sparseMatrix1.add(1, 0, std::complex<double>(0, 2));
	sparseMatrix1.add(1, 1, 3);
	sparseMatrix1.constructCSX();

	//Real vector.
	Matrix<double> b0(2, 1);

	//Complex vector.
	Matrix<std::complex<double>> b1(2, 1);

	//Real matrix, real vector.
	solver.setMatrix(sparseMatrix0);
	b0.at(0, 0) = 2;
	b0.at(1, 0) = 1;
	solver.solve(b0);
	EXPECT_DOUBLE_EQ(b0.at(0, 0), -3);
	EXPECT_DOUBLE_EQ(b0.at(1, 0), 2.5);

	//Real matrix, complex vector.
	b1.at(0, 0) = 1;
	b1.at(1, 0) = std::complex<double>(0, 1);
	solver.solve(b1);
	EXPECT_DOUBLE_EQ(real(b1.at(0, 0)), -2);
	EXPECT_DOUBLE_EQ(imag(b1.at(0, 0)), 1);
	EXPECT_DOUBLE_EQ(real(b1.at(1, 0)), 1.5);
	EXPECT_DOUBLE_EQ(imag(b1.at(1, 0)), -0.5);

	//Complex matrix, real vector. (Fail because it is not generally
	//possible to solve such a problemand get a real answer)
	solver.setMatrix(sparseMatrix1);
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			solver.solve(b0);
		},
		::testing::ExitedWithCode(1),
		""
	);
	::testing::FLAGS_gtest_death_test_style = "fast";

	//Complex matrix, complex vector.
	b1.at(0, 0) = 1;
	b1.at(1, 0) = std::complex<double>(0, 1);
	solver.solve(b1);
	EXPECT_DOUBLE_EQ(real(b1.at(0, 0)), 0.8);
	EXPECT_DOUBLE_EQ(imag(b1.at(0, 0)), 0);
	EXPECT_DOUBLE_EQ(real(b1.at(1, 0)), 0);
	EXPECT_DOUBLE_EQ(imag(b1.at(1, 0)), -0.2);
}

};	//End of namespace Solver
};	//End of namespace TBTK
