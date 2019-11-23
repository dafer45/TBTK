#include "TBTK/ParallelepipedCell.h"
#include "TBTK/Vector2d.h"

#include "gtest/gtest.h"

namespace TBTK{

class ParallelepipedCellTest : public ::testing::Test{
protected:
	ParallelepipedCell parallelepipedCell[6];

	//Small parameter used to set the displacement away from the corners.
	double d = 1e-5;

	//The 2D Wigner-Seitz cell used in the test is a hexagon with the
	//following corners.
	std::vector<std::vector<double>> corners2D;
	//Displacements from the corners that take the points onto the first,
	//second, second, and third Wigner-Seitz cell, respectively. These
	//points are used to check that the division into Wigner-Seitz cell is
	//properly done around the corners.
	std::vector<std::vector<std::vector<double>>> displacements2D;
	std::vector<std::vector<Index>> expectedCells2D;

	//The 3D Wigner-Seitz cell used in the test is a hexagon with the
	//following corners.
	std::vector<std::vector<double>> corners3D;
	//Displacements from the corners that take the points onto the first,
	//second, second, and third Wigner-Seitz cell, respectively. These
	//points are used to check that the division into Wigner-Seitz cell is
	//properly done around the corners.
	std::vector<std::vector<std::vector<double>>> displacements3D;
	std::vector<std::vector<Index>> expectedCells3D;

	void SetUp() override{
		//Set up 1D, 2D, and 3D Wigner-Seitz cell with Nodal and
		//Interior mesh types.
		parallelepipedCell[0] = ParallelepipedCell(
			{{2}},
			SpacePartition::MeshType::Nodal
		);
		parallelepipedCell[1] = ParallelepipedCell(
			{{2}},
			SpacePartition::MeshType::Interior
		);
		parallelepipedCell[2] = ParallelepipedCell(
			{{2, 0}, {-1, sqrt(3)}},
			SpacePartition::MeshType::Nodal
		);
		parallelepipedCell[3] = ParallelepipedCell(
			{{2, 0}, {-1, sqrt(3)}},
			SpacePartition::MeshType::Interior
		);
		parallelepipedCell[4] = ParallelepipedCell(
			{
				{2, 0, 0},
				{-1, sqrt(3), 0},
				{0, 0, 2}
			},
			SpacePartition::MeshType::Nodal
		);
		parallelepipedCell[5] = ParallelepipedCell(
			{
				{2, 0, 0},
				{-1, sqrt(3), 0},
				{0, 0, 2}
			},
			SpacePartition::MeshType::Interior
		);

		//Corners of the first parallelepiped.
		corners2D = {
			{1/2., sqrt(3)/2},
			{-3/2., sqrt(3)/2},
			{-1/2., -sqrt(3)/2},
			{3/2., -sqrt(3)/2}
		};
		//Displacements away from the corners above and into the
		//interior of the neighboring cells. Each group of four
		//correspond to one point in corners2D.
		displacements2D = {
			{
				{-d, -d},
				{d, -d},
				{d, d},
				{-d, d}
			},
			{
				{d, -d},
				{d, d},
				{-d, d},
				{-d, -d}
			},
			{
				{d, d},
				{-d, d},
				{-d, -d},
				{d, -d}
			},
			{
				{-d, d},
				{-d, -d},
				{d, -d},
				{d, d}
			}
		};
		expectedCells2D = {
			{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			{{0, 0}, {0, 1}, {-1, 1}, {-1, 0}},
			{{0, 0}, {-1, 0}, {-1, -1}, {0, -1}},
			{{0, 0}, {0, -1}, {1, -1}, {1, 0}}
		};

		//Points that lie at the intersection between different
		//Wigner-Seitz cell for the 3D lattice.
		corners3D = {
			{1/2., sqrt(3)/2, 1},
			{-3/2., sqrt(3)/2, 1},
			{-1/2., -sqrt(3)/2, 1},
			{3/2., -sqrt(3)/2, 1},
			{1/2., sqrt(3)/2, -1},
			{-3/2., sqrt(3)/2, -1},
			{-1/2., -sqrt(3)/2, -1},
			{3/2., -sqrt(3)/2, -1}
		};
		//Displacements away from the points above and into the
		//interior of the 0th, 1st, 2nd, 1st, 2nd, 3rd, 1st, 2nd, 3rd,
		//2nd, and 5th Wigner-Seitz cell, respectively. Each group of
		//eleven corresponds to one point in corners3D. The second to
		//sixth group is a rotation of the first group by pi/3, 2pi/3,
		//... 5pi/3. The last six are reflections of the first six
		//through the x/y-plane.
		displacements3D = {
			{
				{-d, -d, -d},
				{-d, -d, d},
				{d, -d, -d},
				{d, -d, d},
				{d, d, -d},
				{d, d, d},
				{-d, d, -d},
				{-d, d, d}
			},
			{
				{d, -d, -d},
				{d, -d, d},
				{d, d, -d},
				{d, d, d},
				{-d, d, -d},
				{-d, d, d},
				{-d, -d, -d},
				{-d, -d, d}
			},
			{
				{d, d, -d},
				{d, d, d},
				{-d, d, -d},
				{-d, d, d},
				{-d, -d, -d},
				{-d, -d, d},
				{d, -d, -d},
				{d, -d, d}
			},
			{
				{-d, d, -d},
				{-d, d, d},
				{-d, -d, -d},
				{-d, -d, d},
				{d, -d, -d},
				{d, -d, d},
				{d, d, -d},
				{d, d, d}
			},
			{
				{-d, -d, d},
				{-d, -d, -d},
				{d, -d, d},
				{d, -d, -d},
				{d, d, d},
				{d, d, -d},
				{-d, d, d},
				{-d, d, -d}
			},
			{
				{d, -d, d},
				{d, -d, -d},
				{d, d, d},
				{d, d, -d},
				{-d, d, d},
				{-d, d, -d},
				{-d, -d, d},
				{-d, -d, -d}
			},
			{
				{d, d, d},
				{d, d, -d},
				{-d, d, d},
				{-d, d, -d},
				{-d, -d, d},
				{-d, -d, -d},
				{d, -d, d},
				{d, -d, -d}
			},
			{
				{-d, d, d},
				{-d, d, -d},
				{-d, -d, d},
				{-d, -d, -d},
				{d, -d, d},
				{d, -d, -d},
				{d, d, d},
				{d, d, -d}
			}
		};
		expectedCells3D = {
			{
				{0, 0, 0},
				{0, 0, 1},
				{1, 0, 0},
				{1, 0, 1},
				{1, 1, 0},
				{1, 1, 1},
				{0, 1, 0},
				{0, 1, 1}
			},
			{
				{0, 0, 0},
				{0, 0, 1},
				{0, 1, 0},
				{0, 1, 1},
				{-1, 1, 0},
				{-1, 1, 1},
				{-1, 0, 0},
				{-1, 0, 1}
			},
			{
				{0, 0, 0},
				{0, 0, 1},
				{-1, 0, 0},
				{-1, 0, 1},
				{-1, -1, 0},
				{-1, -1, 1},
				{0, -1, 0},
				{0, -1, 1}
			},
			{
				{0, 0, 0},
				{0, 0, 1},
				{0, -1, 0},
				{0, -1, 1},
				{1, -1, 0},
				{1, -1, 1},
				{1, 0, 0},
				{1, 0, 1}
			},
			{
				{0, 0, 0},
				{0, 0, -1},
				{1, 0, 0},
				{1, 0, -1},
				{1, 1, 0},
				{1, 1, -1},
				{0, 1, 0},
				{0, 1, -1}
			},
			{
				{0, 0, 0},
				{0, 0, -1},
				{0, 1, 0},
				{0, 1, -1},
				{-1, 1, 0},
				{-1, 1, -1},
				{-1, 0, 0},
				{-1, 0, -1}
			},
			{
				{0, 0, 0},
				{0, 0, -1},
				{-1, 0, 0},
				{-1, 0, -1},
				{-1, -1, 0},
				{-1, -1, -1},
				{0, -1, 0},
				{0, -1, -1}
			},
			{
				{0, 0, 0},
				{0, 0, -1},
				{0, -1, 0},
				{0, -1, -1},
				{1, -1, 0},
				{1, -1, -1},
				{1, 0, 0},
				{1, 0, -1}
			},
		};
	}
};

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorCellIndex.1 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex1){
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({-3 - d}).equals({-2}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({-3 + d}).equals({-1}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({-1 - d}).equals({-1}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({-1 + d}).equals({0}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({1 - d}).equals({0}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({1 + d}).equals({1}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({3 - d}).equals({1}));
	EXPECT_TRUE(parallelepipedCell[0].getMajorCellIndex({3 + d}).equals({2}));
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorCellIndex.2 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex2){
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({-3 - d}).equals({-2}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({-3 + d}).equals({-1}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({-1 - d}).equals({-1}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({-1 + d}).equals({0}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({1 - d}).equals({0}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({1 + d}).equals({1}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({3 - d}).equals({1}));
	EXPECT_TRUE(parallelepipedCell[1].getMajorCellIndex({3 + d}).equals({2}));
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.3 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex3){
	for(unsigned int n = 0; n < corners2D.size(); n++){
		for(unsigned int d = 0; d < displacements2D[n].size(); d++){
			std::vector<double> coordinate = {
				corners2D[n][0] + displacements2D[n][d][0],
				corners2D[n][1] + displacements2D[n][d][1]
			};
			EXPECT_TRUE(
				parallelepipedCell[2].getMajorCellIndex(
					coordinate
				).equals(expectedCells2D[n][d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.4 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex4){
	for(unsigned int n = 0; n < corners2D.size(); n++){
		for(unsigned int d = 0; d < displacements2D[n].size(); d++){
			std::vector<double> coordinate = {
				corners2D[n][0] + displacements2D[n][d][0],
				corners2D[n][1] + displacements2D[n][d][1]
			};
			EXPECT_TRUE(
				parallelepipedCell[3].getMajorCellIndex(
					coordinate
				).equals(expectedCells2D[n][d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.5 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex5){
	for(unsigned int n = 0; n < corners3D.size(); n++){
		for(unsigned int d = 0; d < displacements3D[n].size(); d++){
			std::vector<double> coordinate = {
				corners3D[n][0] + displacements3D[n][d][0],
				corners3D[n][1] + displacements3D[n][d][1],
				corners3D[n][2] + displacements3D[n][d][2]
			};
			EXPECT_TRUE(
				parallelepipedCell[4].getMajorCellIndex(
					coordinate
				).equals(expectedCells3D[n][d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.6 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex6){
	for(unsigned int n = 0; n < corners3D.size(); n++){
		for(unsigned int d = 0; d < displacements3D[n].size(); d++){
			std::vector<double> coordinate = {
				corners3D[n][0] + displacements3D[n][d][0],
				corners3D[n][1] + displacements3D[n][d][1],
				corners3D[n][2] + displacements3D[n][d][2]
			};
			EXPECT_TRUE(
				parallelepipedCell[5].getMajorCellIndex(
					coordinate
				).equals(expectedCells3D[n][d])
			);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorCellIndex.7 2019-11-21
TEST_F(ParallelepipedCellTest, getMajorCellIndex7){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			parallelepipedCell[0].getMajorCellIndex({1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.1 2019-11-21
TEST_F(ParallelepipedCellTest, getMinorCellIndex1){
	std::vector<std::vector<double>> points = {
		{-4/3 - d},
		{-4/3 + d},
		{-1/3. - d},
		{-1/3. + d},
		{1/3. - d},
		{1/3. + d},
		{4/3 - d},
		{4/3 + d}
	};
	std::vector<Index> expectedMinorCells = {{-2}, {-1}, {-1}, {0}, {0}, {1}, {1}, {2}};
	for(unsigned int n = 0; n < points.size(); n++){
		EXPECT_TRUE(
			parallelepipedCell[0].getMinorCellIndex(
				points[n],
				{3}
			).equals(expectedMinorCells[n])
		);
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.2 2019-11-21
TEST_F(ParallelepipedCellTest, getMinorCellIndex2){
	std::vector<std::vector<double>> points = {
		{-2/3. - d},
		{-2/3. + d},
		{-d},
		{d},
		{2/3. - d},
		{2/3. + d},
		{4/3. - d},
		{4/3. + d}
	};
	std::vector<Index> expectedMinorCells = {{-2}, {-1}, {-1}, {0}, {0}, {1}, {1}, {2}};
	for(unsigned int n = 0; n < points.size(); n++){
		EXPECT_TRUE(
			parallelepipedCell[1].getMinorCellIndex(
				points[n],
				{3}
			).equals(expectedMinorCells[n])
		);
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.3 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorCellIndex3){
	Vector2d u({2/3., 0});
	Vector2d v({-1/2., sqrt(3)/2});

	std::vector<std::vector<double>> points = {
		{(u+v).x/2. - d, (u+v).y/2. - d},
		{(u+v).x/2. + d, (u+v).y/2. - d},
		{(u+v).x/2. + d, (u+v).y/2. + d},
		{(u+v).x/2. - d, (u+v).y/2. + d},
		{(-u+v).x/2. + d, (-u+v).y/2. - d},
		{(-u+v).x/2. + d, (-u+v).y/2. + d},
		{(-u+v).x/2. - d, (-u+v).y/2. + d},
		{(-u+v).x/2. - d, (-u+v).y/2. - d},
		{-(u+v).x/2. + d, -(u+v).y/2. + d},
		{-(u+v).x/2. - d, -(u+v).y/2. + d},
		{-(u+v).x/2. - d, -(u+v).y/2. - d},
		{-(u+v).x/2. + d, -(u+v).y/2. - d},
		{(u-v).x/2. - d, (u-v).y/2. + d},
		{(u-v).x/2. - d, (u-v).y/2. - d},
		{(u-v).x/2. + d, (u-v).y/2. - d},
		{(u-v).x/2. + d, (u-v).y/2. + d},
		{(5*u+7*v).x/2. - d, (5*u+7*v).y/2. - d},
		{(5*u+7*v).x/2. + d, (5*u+7*v).y/2. - d},
		{(5*u+7*v).x/2. + d, (5*u+7*v).y/2. + d},
		{(5*u+7*v).x/2. - d, (5*u+7*v).y/2. + d},
		{(-5*u+7*v).x/2. + d, (-5*u+7*v).y/2. - d},
		{(-5*u+7*v).x/2. + d, (-5*u+7*v).y/2. + d},
		{(-5*u+7*v).x/2. - d, (-5*u+7*v).y/2. + d},
		{(-5*u+7*v).x/2. - d, (-5*u+7*v).y/2. - d},
		{-(5*u+7*v).x/2. + d, -(5*u+7*v).y/2. + d},
		{-(5*u+7*v).x/2. - d, -(5*u+7*v).y/2. + d},
		{-(5*u+7*v).x/2. - d, -(5*u+7*v).y/2. - d},
		{-(5*u+7*v).x/2. + d, -(5*u+7*v).y/2. - d},
		{(5*u-7*v).x/2. - d, (5*u-7*v).y/2. + d},
		{(5*u-7*v).x/2. - d, (5*u-7*v).y/2. - d},
		{(5*u-7*v).x/2. + d, (5*u-7*v).y/2. - d},
		{(5*u-7*v).x/2. + d, (5*u-7*v).y/2. + d}
	};
	std::vector<Index> expectedMinorCells = {
		{0, 0},
		{1, 0},
		{1, 1},
		{0, 1},
		{0, 0},
		{0, 1},
		{-1, 1},
		{-1, 0},
		{0, 0},
		{-1, 0},
		{-1, -1},
		{0, -1},
		{0, 0},
		{0, -1},
		{1, -1},
		{1, 0},
		{2, 3},
		{3, 3},
		{3, 4},
		{2, 4},
		{-2, 3},
		{-2, 4},
		{-3, 4},
		{-3, 3},
		{-2, -3},
		{-3, -3},
		{-3, -4},
		{-2, -4},
		{2, -3},
		{2, -4},
		{3, -4},
		{3, -3}
	};
	for(unsigned int n = 0; n < points.size(); n++){
		EXPECT_TRUE(
			parallelepipedCell[2].getMinorCellIndex(
				points[n],
				{3, 2}
			).equals(expectedMinorCells[n])
		);
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.4 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorCellIndex4){
	Vector2d u({2/3., 0});
	Vector2d v({-1/2., sqrt(3)/2});

	std::vector<std::vector<double>> points = {
		{d, d},
		{-d, d},
		{-d, -d},
		{d, -d},
		{(5*u+7*v).x + d, (5*u+7*v).y + d},
		{(5*u+7*v).x - d, (5*u+7*v).y + d},
		{(5*u+7*v).x - d, (5*u+7*v).y - d},
		{(5*u+7*v).x + d, (5*u+7*v).y - d},
		{(-5*u+7*v).x + d, (-5*u+7*v).y + d},
		{(-5*u+7*v).x - d, (-5*u+7*v).y + d},
		{(-5*u+7*v).x - d, (-5*u+7*v).y - d},
		{(-5*u+7*v).x + d, (-5*u+7*v).y - d},
		{(-5*u-7*v).x + d, (-5*u-7*v).y + d},
		{(-5*u-7*v).x - d, (-5*u-7*v).y + d},
		{(-5*u-7*v).x - d, (-5*u-7*v).y - d},
		{(-5*u-7*v).x + d, (-5*u-7*v).y - d},
		{(5*u-7*v).x + d, (5*u-7*v).y + d},
		{(5*u-7*v).x - d, (5*u-7*v).y + d},
		{(5*u-7*v).x - d, (5*u-7*v).y - d},
		{(5*u-7*v).x + d, (5*u-7*v).y - d}
	};
	std::vector<Index> expectedMinorCells = {
		{0, 0},
		{-1, 0},
		{-1, -1},
		{0, -1},
		{5, 7},
		{4, 7},
		{4, 6},
		{5, 6},
		{-5, 7},
		{-6, 7},
		{-6, 6},
		{-5, 6},
		{-5, -7},
		{-6, -7},
		{-6, -8},
		{-5, -8},
		{5, -7},
		{4, -7},
		{4, -8},
		{5, -8}
	};
	for(unsigned int n = 0; n < points.size(); n++){
		EXPECT_TRUE(
			parallelepipedCell[3].getMinorCellIndex(
				points[n],
				{3, 2}
			).equals(expectedMinorCells[n])
		);
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.5 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorCellIndex5){
	Vector3d u({2/3., 0, 0});
	Vector3d v({-1/2., sqrt(3)/2, 0});
	Vector3d w({0, 0, 1/2.});

	std::vector<std::vector<double>> points = {
		{(u+v-w).x/2. - d, (u+v+w).y/2. - d, (u+v+w).z/2. - d},
		{(u+v-w).x/2. + d, (u+v+w).y/2. - d, (u+v+w).z/2. - d},
		{(u+v-w).x/2. + d, (u+v+w).y/2. + d, (u+v+w).z/2. - d},
		{(u+v-w).x/2. - d, (u+v+w).y/2. + d, (u+v+w).z/2. - d},
		{(u+v-w).x/2. - d, (u+v+w).y/2. - d, (u+v+w).z/2. + d},
		{(u+v-w).x/2. + d, (u+v+w).y/2. - d, (u+v+w).z/2. + d},
		{(u+v-w).x/2. + d, (u+v+w).y/2. + d, (u+v+w).z/2. + d},
		{(u+v-w).x/2. - d, (u+v+w).y/2. + d, (u+v+w).z/2. + d},
		{(-u+v-w).x/2. + d, (-u+v+w).y/2. - d, (-u+v+w).z/2. - d},
		{(-u+v-w).x/2. + d, (-u+v+w).y/2. + d, (-u+v+w).z/2. - d},
		{(-u+v-w).x/2. - d, (-u+v+w).y/2. + d, (-u+v+w).z/2. - d},
		{(-u+v-w).x/2. - d, (-u+v+w).y/2. - d, (-u+v+w).z/2. - d},
		{(-u+v-w).x/2. + d, (-u+v+w).y/2. - d, (-u+v+w).z/2. + d},
		{(-u+v-w).x/2. + d, (-u+v+w).y/2. + d, (-u+v+w).z/2. + d},
		{(-u+v-w).x/2. - d, (-u+v+w).y/2. + d, (-u+v+w).z/2. + d},
		{(-u+v-w).x/2. - d, (-u+v+w).y/2. - d, (-u+v+w).z/2. + d},
		{(-u-v-w).x/2. + d, (-u-v+w).y/2. + d, (-u-v+w).z/2. - d},
		{(-u-v-w).x/2. - d, (-u-v+w).y/2. + d, (-u-v+w).z/2. - d},
		{(-u-v-w).x/2. - d, (-u-v+w).y/2. - d, (-u-v+w).z/2. - d},
		{(-u-v-w).x/2. + d, (-u-v+w).y/2. - d, (-u-v+w).z/2. - d},
		{(-u-v-w).x/2. + d, (-u-v+w).y/2. + d, (-u-v+w).z/2. + d},
		{(-u-v-w).x/2. - d, (-u-v+w).y/2. + d, (-u-v+w).z/2. + d},
		{(-u-v-w).x/2. - d, (-u-v+w).y/2. - d, (-u-v+w).z/2. + d},
		{(-u-v-w).x/2. + d, (-u-v+w).y/2. - d, (-u-v+w).z/2. + d},
		{(u-v-w).x/2. - d, (u-v+w).y/2. + d, (u-v+w).z/2. - d},
		{(u-v-w).x/2. - d, (u-v+w).y/2. - d, (u-v+w).z/2. - d},
		{(u-v-w).x/2. + d, (u-v+w).y/2. - d, (u-v+w).z/2. - d},
		{(u-v-w).x/2. + d, (u-v+w).y/2. + d, (u-v+w).z/2. - d},
		{(u-v-w).x/2. - d, (u-v+w).y/2. + d, (u-v+w).z/2. + d},
		{(u-v-w).x/2. - d, (u-v+w).y/2. - d, (u-v+w).z/2. + d},
		{(u-v-w).x/2. + d, (u-v+w).y/2. - d, (u-v+w).z/2. + d},
		{(u-v-w).x/2. + d, (u-v+w).y/2. + d, (u-v+w).z/2. + d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v+9*w).y/2. - d, (5*u+7*v+9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v+9*w).y/2. - d, (5*u+7*v+9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v+9*w).y/2. + d, (5*u+7*v+9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v+9*w).y/2. + d, (5*u+7*v+9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v+9*w).y/2. - d, (5*u+7*v+9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v+9*w).y/2. - d, (5*u+7*v+9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v+9*w).y/2. + d, (5*u+7*v+9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v+9*w).y/2. + d, (5*u+7*v+9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v+9*w).y/2. - d, (-5*u+7*v+9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v+9*w).y/2. + d, (-5*u+7*v+9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v+9*w).y/2. + d, (-5*u+7*v+9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v+9*w).y/2. - d, (-5*u+7*v+9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v+9*w).y/2. - d, (-5*u+7*v+9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v+9*w).y/2. + d, (-5*u+7*v+9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v+9*w).y/2. + d, (-5*u+7*v+9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v+9*w).y/2. - d, (-5*u+7*v+9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v+9*w).y/2. + d, (-5*u-7*v+9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v+9*w).y/2. + d, (-5*u-7*v+9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v+9*w).y/2. - d, (-5*u-7*v+9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v+9*w).y/2. - d, (-5*u-7*v+9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v+9*w).y/2. + d, (-5*u-7*v+9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v+9*w).y/2. + d, (-5*u-7*v+9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v+9*w).y/2. - d, (-5*u-7*v+9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v+9*w).y/2. - d, (-5*u-7*v+9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v+9*w).y/2. + d, (5*u-7*v+9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v+9*w).y/2. - d, (5*u-7*v+9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v+9*w).y/2. - d, (5*u-7*v+9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v+9*w).y/2. + d, (5*u-7*v+9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v+9*w).y/2. + d, (5*u-7*v+9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v+9*w).y/2. - d, (5*u-7*v+9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v+9*w).y/2. - d, (5*u-7*v+9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v+9*w).y/2. + d, (5*u-7*v+9*w).z/2. + d},
		{(u+v-w).x/2. - d, (u+v-w).y/2. - d, (u+v-w).z/2. + d},
		{(u+v-w).x/2. + d, (u+v-w).y/2. - d, (u+v-w).z/2. + d},
		{(u+v-w).x/2. + d, (u+v-w).y/2. + d, (u+v-w).z/2. + d},
		{(u+v-w).x/2. - d, (u+v-w).y/2. + d, (u+v-w).z/2. + d},
		{(u+v-w).x/2. - d, (u+v-w).y/2. - d, (u+v-w).z/2. - d},
		{(u+v-w).x/2. + d, (u+v-w).y/2. - d, (u+v-w).z/2. - d},
		{(u+v-w).x/2. + d, (u+v-w).y/2. + d, (u+v-w).z/2. - d},
		{(u+v-w).x/2. - d, (u+v-w).y/2. + d, (u+v-w).z/2. - d},
		{(-u+v-w).x/2. + d, (-u+v-w).y/2. - d, (-u+v-w).z/2. + d},
		{(-u+v-w).x/2. + d, (-u+v-w).y/2. + d, (-u+v-w).z/2. + d},
		{(-u+v-w).x/2. - d, (-u+v-w).y/2. + d, (-u+v-w).z/2. + d},
		{(-u+v-w).x/2. - d, (-u+v-w).y/2. - d, (-u+v-w).z/2. + d},
		{(-u+v-w).x/2. + d, (-u+v-w).y/2. - d, (-u+v-w).z/2. - d},
		{(-u+v-w).x/2. + d, (-u+v-w).y/2. + d, (-u+v-w).z/2. - d},
		{(-u+v-w).x/2. - d, (-u+v-w).y/2. + d, (-u+v-w).z/2. - d},
		{(-u+v-w).x/2. - d, (-u+v-w).y/2. - d, (-u+v-w).z/2. - d},
		{(-u-v-w).x/2. + d, (-u-v-w).y/2. + d, (-u-v-w).z/2. + d},
		{(-u-v-w).x/2. - d, (-u-v-w).y/2. + d, (-u-v-w).z/2. + d},
		{(-u-v-w).x/2. - d, (-u-v-w).y/2. - d, (-u-v-w).z/2. + d},
		{(-u-v-w).x/2. + d, (-u-v-w).y/2. - d, (-u-v-w).z/2. + d},
		{(-u-v-w).x/2. + d, (-u-v-w).y/2. + d, (-u-v-w).z/2. - d},
		{(-u-v-w).x/2. - d, (-u-v-w).y/2. + d, (-u-v-w).z/2. - d},
		{(-u-v-w).x/2. - d, (-u-v-w).y/2. - d, (-u-v-w).z/2. - d},
		{(-u-v-w).x/2. + d, (-u-v-w).y/2. - d, (-u-v-w).z/2. - d},
		{(u-v-w).x/2. - d, (u-v-w).y/2. + d, (u-v-w).z/2. + d},
		{(u-v-w).x/2. - d, (u-v-w).y/2. - d, (u-v-w).z/2. + d},
		{(u-v-w).x/2. + d, (u-v-w).y/2. - d, (u-v-w).z/2. + d},
		{(u-v-w).x/2. + d, (u-v-w).y/2. + d, (u-v-w).z/2. + d},
		{(u-v-w).x/2. - d, (u-v-w).y/2. + d, (u-v-w).z/2. - d},
		{(u-v-w).x/2. - d, (u-v-w).y/2. - d, (u-v-w).z/2. - d},
		{(u-v-w).x/2. + d, (u-v-w).y/2. - d, (u-v-w).z/2. - d},
		{(u-v-w).x/2. + d, (u-v-w).y/2. + d, (u-v-w).z/2. - d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v-9*w).y/2. - d, (5*u+7*v-9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v-9*w).y/2. - d, (5*u+7*v-9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v-9*w).y/2. + d, (5*u+7*v-9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v-9*w).y/2. + d, (5*u+7*v-9*w).z/2. + d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v-9*w).y/2. - d, (5*u+7*v-9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v-9*w).y/2. - d, (5*u+7*v-9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. + d, (5*u+7*v-9*w).y/2. + d, (5*u+7*v-9*w).z/2. - d},
		{(5*u+7*v-9*w).x/2. - d, (5*u+7*v-9*w).y/2. + d, (5*u+7*v-9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v-9*w).y/2. - d, (-5*u+7*v-9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v-9*w).y/2. + d, (-5*u+7*v-9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v-9*w).y/2. + d, (-5*u+7*v-9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v-9*w).y/2. - d, (-5*u+7*v-9*w).z/2. + d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v-9*w).y/2. - d, (-5*u+7*v-9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. + d, (-5*u+7*v-9*w).y/2. + d, (-5*u+7*v-9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v-9*w).y/2. + d, (-5*u+7*v-9*w).z/2. - d},
		{(-5*u+7*v-9*w).x/2. - d, (-5*u+7*v-9*w).y/2. - d, (-5*u+7*v-9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v-9*w).y/2. + d, (-5*u-7*v-9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v-9*w).y/2. + d, (-5*u-7*v-9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v-9*w).y/2. - d, (-5*u-7*v-9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v-9*w).y/2. - d, (-5*u-7*v-9*w).z/2. + d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v-9*w).y/2. + d, (-5*u-7*v-9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v-9*w).y/2. + d, (-5*u-7*v-9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. - d, (-5*u-7*v-9*w).y/2. - d, (-5*u-7*v-9*w).z/2. - d},
		{(-5*u-7*v-9*w).x/2. + d, (-5*u-7*v-9*w).y/2. - d, (-5*u-7*v-9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v-9*w).y/2. + d, (5*u-7*v-9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v-9*w).y/2. - d, (5*u-7*v-9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v-9*w).y/2. - d, (5*u-7*v-9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v-9*w).y/2. + d, (5*u-7*v-9*w).z/2. + d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v-9*w).y/2. + d, (5*u-7*v-9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. - d, (5*u-7*v-9*w).y/2. - d, (5*u-7*v-9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v-9*w).y/2. - d, (5*u-7*v-9*w).z/2. - d},
		{(5*u-7*v-9*w).x/2. + d, (5*u-7*v-9*w).y/2. + d, (5*u-7*v-9*w).z/2. - d}
	};
	std::vector<Index> expectedMinorCells = {
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1},
		{0, 1, 1},
		{0, 0, 0},
		{0, 1, 0},
		{-1, 1, 0},
		{-1, 0, 0},
		{0, 0, 1},
		{0, 1, 1},
		{-1, 1, 1},
		{-1, 0, 1},
		{0, 0, 0},
		{-1, 0, 0},
		{-1, -1, 0},
		{0, -1, 0},
		{0, 0, 1},
		{-1, 0, 1},
		{-1, -1, 1},
		{0, -1, 1},
		{0, 0, 0},
		{0, -1, 0},
		{1, -1, 0},
		{1, 0, 0},
		{0, 0, 1},
		{0, -1, 1},
		{1, -1, 1},
		{1, 0, 1},
		{2, 3, 4},
		{3, 3, 4},
		{3, 4, 4},
		{2, 4, 4},
		{2, 3, 5},
		{3, 3, 5},
		{3, 4, 5},
		{2, 4, 5},
		{-2, 3, 4},
		{-2, 4, 4},
		{-3, 4, 4},
		{-3, 3, 4},
		{-2, 3, 5},
		{-2, 4, 5},
		{-3, 4, 5},
		{-3, 3, 5},
		{-2, -3, 4},
		{-3, -3, 4},
		{-3, -4, 4},
		{-2, -4, 4},
		{-2, -3, 5},
		{-3, -3, 5},
		{-3, -4, 5},
		{-2, -4, 5},
		{2, -3, 4},
		{2, -4, 4},
		{3, -4, 4},
		{3, -3, 4},
		{2, -3, 5},
		{2, -4, 5},
		{3, -4, 5},
		{3, -3, 5},
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
		{0, 0, -1},
		{1, 0, -1},
		{1, 1, -1},
		{0, 1, -1},
		{0, 0, 0},
		{0, 1, 0},
		{-1, 1, 0},
		{-1, 0, 0},
		{0, 0, -1},
		{0, 1, -1},
		{-1, 1, -1},
		{-1, 0, -1},
		{0, 0, 0},
		{-1, 0, 0},
		{-1, -1, 0},
		{0, -1, 0},
		{0, 0, -1},
		{-1, 0, -1},
		{-1, -1, -1},
		{0, -1, -1},
		{0, 0, 0},
		{0, -1, 0},
		{1, -1, 0},
		{1, 0, 0},
		{0, 0, -1},
		{0, -1, -1},
		{1, -1, -1},
		{1, 0, -1},
		{2, 3, -4},
		{3, 3, -4},
		{3, 4, -4},
		{2, 4, -4},
		{2, 3, -5},
		{3, 3, -5},
		{3, 4, -5},
		{2, 4, -5},
		{-2, 3, -4},
		{-2, 4, -4},
		{-3, 4, -4},
		{-3, 3, -4},
		{-2, 3, -5},
		{-2, 4, -5},
		{-3, 4, -5},
		{-3, 3, -5},
		{-2, -3, -4},
		{-3, -3, -4},
		{-3, -4, -4},
		{-2, -4, -4},
		{-2, -3, -5},
		{-3, -3, -5},
		{-3, -4, -5},
		{-2, -4, -5},
		{2, -3, -4},
		{2, -4, -4},
		{3, -4, -4},
		{3, -3, -4},
		{2, -3, -5},
		{2, -4, -5},
		{3, -4, -5},
		{3, -3, -5}
	};
	for(unsigned int n = 0; n < points.size(); n++){
		EXPECT_TRUE(
			parallelepipedCell[4].getMinorCellIndex(
				points[n],
				{3, 2, 4}
			).equals(expectedMinorCells[n])
		);
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.6 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorCellIndex6){
	Vector3d u({2/3., 0, 0});
	Vector3d v({-1/2., sqrt(3)/2, 0});
	Vector3d w({0, 0, 1/2.});

	std::vector<std::vector<double>> points = {
		{d, d, d},
		{-d, d, d},
		{-d, -d, d},
		{d, -d, d},
		{d, d, -d},
		{-d, d, -d},
		{-d, -d, -d},
		{d, -d, -d},
		{(5*u+7*v+9*w).x + d, (5*u+7*v+9*w).y + d, (5*u+7*v+9*w).z + d},
		{(5*u+7*v+9*w).x - d, (5*u+7*v+9*w).y + d, (5*u+7*v+9*w).z + d},
		{(5*u+7*v+9*w).x - d, (5*u+7*v+9*w).y - d, (5*u+7*v+9*w).z + d},
		{(5*u+7*v+9*w).x + d, (5*u+7*v+9*w).y - d, (5*u+7*v+9*w).z + d},
		{(5*u+7*v+9*w).x + d, (5*u+7*v+9*w).y + d, (5*u+7*v+9*w).z - d},
		{(5*u+7*v+9*w).x - d, (5*u+7*v+9*w).y + d, (5*u+7*v+9*w).z - d},
		{(5*u+7*v+9*w).x - d, (5*u+7*v+9*w).y - d, (5*u+7*v+9*w).z - d},
		{(5*u+7*v+9*w).x + d, (5*u+7*v+9*w).y - d, (5*u+7*v+9*w).z - d},
		{(-5*u+7*v+9*w).x + d, (-5*u+7*v+9*w).y + d, (-5*u+7*v+9*w).z + d},
		{(-5*u+7*v+9*w).x - d, (-5*u+7*v+9*w).y + d, (-5*u+7*v+9*w).z + d},
		{(-5*u+7*v+9*w).x - d, (-5*u+7*v+9*w).y - d, (-5*u+7*v+9*w).z + d},
		{(-5*u+7*v+9*w).x + d, (-5*u+7*v+9*w).y - d, (-5*u+7*v+9*w).z + d},
		{(-5*u+7*v+9*w).x + d, (-5*u+7*v+9*w).y + d, (-5*u+7*v+9*w).z - d},
		{(-5*u+7*v+9*w).x - d, (-5*u+7*v+9*w).y + d, (-5*u+7*v+9*w).z - d},
		{(-5*u+7*v+9*w).x - d, (-5*u+7*v+9*w).y - d, (-5*u+7*v+9*w).z - d},
		{(-5*u+7*v+9*w).x + d, (-5*u+7*v+9*w).y - d, (-5*u+7*v+9*w).z - d},
		{(-5*u-7*v+9*w).x + d, (-5*u-7*v+9*w).y + d, (-5*u-7*v+9*w).z + d},
		{(-5*u-7*v+9*w).x - d, (-5*u-7*v+9*w).y + d, (-5*u-7*v+9*w).z + d},
		{(-5*u-7*v+9*w).x - d, (-5*u-7*v+9*w).y - d, (-5*u-7*v+9*w).z + d},
		{(-5*u-7*v+9*w).x + d, (-5*u-7*v+9*w).y - d, (-5*u-7*v+9*w).z + d},
		{(-5*u-7*v+9*w).x + d, (-5*u-7*v+9*w).y + d, (-5*u-7*v+9*w).z - d},
		{(-5*u-7*v+9*w).x - d, (-5*u-7*v+9*w).y + d, (-5*u-7*v+9*w).z - d},
		{(-5*u-7*v+9*w).x - d, (-5*u-7*v+9*w).y - d, (-5*u-7*v+9*w).z - d},
		{(-5*u-7*v+9*w).x + d, (-5*u-7*v+9*w).y - d, (-5*u-7*v+9*w).z - d},
		{(5*u-7*v+9*w).x + d, (5*u-7*v+9*w).y + d, (5*u-7*v+9*w).z + d},
		{(5*u-7*v+9*w).x - d, (5*u-7*v+9*w).y + d, (5*u-7*v+9*w).z + d},
		{(5*u-7*v+9*w).x - d, (5*u-7*v+9*w).y - d, (5*u-7*v+9*w).z + d},
		{(5*u-7*v+9*w).x + d, (5*u-7*v+9*w).y - d, (5*u-7*v+9*w).z + d},
		{(5*u-7*v+9*w).x + d, (5*u-7*v+9*w).y + d, (5*u-7*v+9*w).z - d},
		{(5*u-7*v+9*w).x - d, (5*u-7*v+9*w).y + d, (5*u-7*v+9*w).z - d},
		{(5*u-7*v+9*w).x - d, (5*u-7*v+9*w).y - d, (5*u-7*v+9*w).z - d},
		{(5*u-7*v+9*w).x + d, (5*u-7*v+9*w).y - d, (5*u-7*v+9*w).z - d},
		{(5*u+7*v-9*w).x + d, (5*u+7*v-9*w).y + d, (5*u+7*v-9*w).z - d},
		{(5*u+7*v-9*w).x - d, (5*u+7*v-9*w).y + d, (5*u+7*v-9*w).z - d},
		{(5*u+7*v-9*w).x - d, (5*u+7*v-9*w).y - d, (5*u+7*v-9*w).z - d},
		{(5*u+7*v-9*w).x + d, (5*u+7*v-9*w).y - d, (5*u+7*v-9*w).z - d},
		{(5*u+7*v-9*w).x + d, (5*u+7*v-9*w).y + d, (5*u+7*v-9*w).z + d},
		{(5*u+7*v-9*w).x - d, (5*u+7*v-9*w).y + d, (5*u+7*v-9*w).z + d},
		{(5*u+7*v-9*w).x - d, (5*u+7*v-9*w).y - d, (5*u+7*v-9*w).z + d},
		{(5*u+7*v-9*w).x + d, (5*u+7*v-9*w).y - d, (5*u+7*v-9*w).z + d},
		{(-5*u+7*v-9*w).x + d, (-5*u+7*v-9*w).y + d, (-5*u+7*v-9*w).z - d},
		{(-5*u+7*v-9*w).x - d, (-5*u+7*v-9*w).y + d, (-5*u+7*v-9*w).z - d},
		{(-5*u+7*v-9*w).x - d, (-5*u+7*v-9*w).y - d, (-5*u+7*v-9*w).z - d},
		{(-5*u+7*v-9*w).x + d, (-5*u+7*v-9*w).y - d, (-5*u+7*v-9*w).z - d},
		{(-5*u+7*v-9*w).x + d, (-5*u+7*v-9*w).y + d, (-5*u+7*v-9*w).z + d},
		{(-5*u+7*v-9*w).x - d, (-5*u+7*v-9*w).y + d, (-5*u+7*v-9*w).z + d},
		{(-5*u+7*v-9*w).x - d, (-5*u+7*v-9*w).y - d, (-5*u+7*v-9*w).z + d},
		{(-5*u+7*v-9*w).x + d, (-5*u+7*v-9*w).y - d, (-5*u+7*v-9*w).z + d},
		{(-5*u-7*v-9*w).x + d, (-5*u-7*v-9*w).y + d, (-5*u-7*v-9*w).z - d},
		{(-5*u-7*v-9*w).x - d, (-5*u-7*v-9*w).y + d, (-5*u-7*v-9*w).z - d},
		{(-5*u-7*v-9*w).x - d, (-5*u-7*v-9*w).y - d, (-5*u-7*v-9*w).z - d},
		{(-5*u-7*v-9*w).x + d, (-5*u-7*v-9*w).y - d, (-5*u-7*v-9*w).z - d},
		{(-5*u-7*v-9*w).x + d, (-5*u-7*v-9*w).y + d, (-5*u-7*v-9*w).z + d},
		{(-5*u-7*v-9*w).x - d, (-5*u-7*v-9*w).y + d, (-5*u-7*v-9*w).z + d},
		{(-5*u-7*v-9*w).x - d, (-5*u-7*v-9*w).y - d, (-5*u-7*v-9*w).z + d},
		{(-5*u-7*v-9*w).x + d, (-5*u-7*v-9*w).y - d, (-5*u-7*v-9*w).z + d},
		{(5*u-7*v-9*w).x + d, (5*u-7*v-9*w).y + d, (5*u-7*v-9*w).z - d},
		{(5*u-7*v-9*w).x - d, (5*u-7*v-9*w).y + d, (5*u-7*v-9*w).z - d},
		{(5*u-7*v-9*w).x - d, (5*u-7*v-9*w).y - d, (5*u-7*v-9*w).z - d},
		{(5*u-7*v-9*w).x + d, (5*u-7*v-9*w).y - d, (5*u-7*v-9*w).z - d},
		{(5*u-7*v-9*w).x + d, (5*u-7*v-9*w).y + d, (5*u-7*v-9*w).z + d},
		{(5*u-7*v-9*w).x - d, (5*u-7*v-9*w).y + d, (5*u-7*v-9*w).z + d},
		{(5*u-7*v-9*w).x - d, (5*u-7*v-9*w).y - d, (5*u-7*v-9*w).z + d},
		{(5*u-7*v-9*w).x + d, (5*u-7*v-9*w).y - d, (5*u-7*v-9*w).z + d}
	};
	std::vector<Index> expectedMinorCells = {
		{0, 0, 0},
		{-1, 0, 0},
		{-1, -1, 0},
		{0, -1, 0},
		{0, 0, -1},
		{-1, 0, -1},
		{-1, -1, -1},
		{0, -1, -1},
		{5, 7, 9},
		{4, 7, 9},
		{4, 6, 9},
		{5, 6, 9},
		{5, 7, 8},
		{4, 7, 8},
		{4, 6, 8},
		{5, 6, 8},
		{-5, 7, 9},
		{-6, 7, 9},
		{-6, 6, 9},
		{-5, 6, 9},
		{-5, 7, 8},
		{-6, 7, 8},
		{-6, 6, 8},
		{-5, 6, 8},
		{-5, -7, 9},
		{-6, -7, 9},
		{-6, -8, 9},
		{-5, -8, 9},
		{-5, -7, 8},
		{-6, -7, 8},
		{-6, -8, 8},
		{-5, -8, 8},
		{5, -7, 9},
		{4, -7, 9},
		{4, -8, 9},
		{5, -8, 9},
		{5, -7, 8},
		{4, -7, 8},
		{4, -8, 8},
		{5, -8, 8},

		{5, 7, -10},
		{4, 7, -10},
		{4, 6, -10},
		{5, 6, -10},
		{5, 7, -9},
		{4, 7, -9},
		{4, 6, -9},
		{5, 6, -9},
		{-5, 7, -10},
		{-6, 7, -10},
		{-6, 6, -10},
		{-5, 6, -10},
		{-5, 7, -9},
		{-6, 7, -9},
		{-6, 6, -9},
		{-5, 6, -9},
		{-5, -7, -10},
		{-6, -7, -10},
		{-6, -8, -10},
		{-5, -8, -10},
		{-5, -7, -9},
		{-6, -7, -9},
		{-6, -8, -9},
		{-5, -8, -9},
		{5, -7, -10},
		{4, -7, -10},
		{4, -8, -10},
		{5, -8, -10},
		{5, -7, -9},
		{4, -7, -9},
		{4, -8, -9},
		{5, -8, -9}
	};
	for(unsigned int n = 0; n < points.size(); n++){
//		Streams::out << n << "\t" << parallelepipedCell[5].getMinorCellIndex(points[n], {3, 2, 4}) << "\n";
		EXPECT_TRUE(
			parallelepipedCell[5].getMinorCellIndex(
				points[n],
				{3, 2, 4}
			).equals(expectedMinorCells[n])
		);
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.7 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorCellIndex7){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			parallelepipedCell[0].getMinorCellIndex(
				{2, 3},	//Wrong length.
				{5}
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorCellIndex.8 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorCellIndex8){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			parallelepipedCell[0].getMinorCellIndex(
				{2},
				{5, 5}	//Wrong length.
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorMesh.1 2019-11-22
TEST_F(ParallelepipedCellTest, getMajorMesh1){
	for(unsigned int n = 0; n < 2; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMajorMesh({3});

		EXPECT_EQ(mesh.size(), 3);
		for(unsigned int c = 0; c < mesh.size(); c++){
			EXPECT_EQ(mesh[c].size(), 1);
			EXPECT_FLOAT_EQ(mesh[c][0], 2*c);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorMesh.2 2019-11-22
TEST_F(ParallelepipedCellTest, getMajorMesh2){
	Vector2d u({2, 0});
	Vector2d v({-1, sqrt(3)});

	for(unsigned int n = 2; n < 4; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMajorMesh({3, 4});

		EXPECT_EQ(mesh.size(), 3*4);
		for(unsigned int c = 0; c < mesh.size(); c++){
			unsigned int x = c/4;
			unsigned int y = c%4;

			EXPECT_EQ(mesh[c].size(), 2);
			EXPECT_FLOAT_EQ(mesh[c][0], (x*u+y*v).x);
			EXPECT_FLOAT_EQ(mesh[c][1], (x*u+y*v).y);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorMesh.3 2019-11-22
TEST_F(ParallelepipedCellTest, getMajorMesh3){
	Vector3d u({2, 0, 0});
	Vector3d v({-1, sqrt(3), 0});
	Vector3d w({0, 0, 2});

	for(unsigned int n = 4; n < 6; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMajorMesh({3, 4, 5});

		EXPECT_EQ(mesh.size(), 3*4*5);
		for(unsigned int c = 0; c < mesh.size(); c++){
			unsigned int x = c/(4*5);
			unsigned int y = (c/5)%4;
			unsigned int z = c%5;

			EXPECT_EQ(mesh[c].size(), 3);
			EXPECT_FLOAT_EQ(mesh[c][0], (x*u+y*v+z*w).x);
			EXPECT_FLOAT_EQ(mesh[c][1], (x*u+y*v+z*w).y);
			EXPECT_FLOAT_EQ(mesh[c][2], (x*u+y*v+z*w).z);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMajorMesh.4 2019-11-22
TEST_F(ParallelepipedCellTest, getMajorMesh4){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			//Wrong argument length.
			parallelepipedCell[0].getMajorMesh({2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMesh.1 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMesh1){
	for(unsigned int n = 0; n < 2; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMinorMesh({3});

		EXPECT_EQ(mesh.size(), 3);
		for(unsigned int c = 0; c < mesh.size(); c++){
			EXPECT_EQ(mesh[c].size(), 1);
			EXPECT_FLOAT_EQ(mesh[c][0], c*(2/3.));
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMesh.2 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMesh2){
	Vector2d u({2, 0});
	Vector2d v({-1, sqrt(3)});

	u = u/3;
	v = v/4;

	for(unsigned int n = 2; n < 4; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMinorMesh({3, 4});

		EXPECT_EQ(mesh.size(), 3*4);
		for(unsigned int c = 0; c < mesh.size(); c++){
			unsigned int x = c/4;
			unsigned int y = c%4;

			EXPECT_EQ(mesh[c].size(), 2);
			EXPECT_FLOAT_EQ(mesh[c][0], (x*u+y*v).x);
			EXPECT_FLOAT_EQ(mesh[c][1], (x*u+y*v).y);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMesh.3 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMesh3){
	Vector3d u({2, 0, 0});
	Vector3d v({-1, sqrt(3), 0});
	Vector3d w({0, 0, 2});

	u = u/3;
	v = v/4;
	w = w/5;

	for(unsigned int n = 4; n < 6; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMinorMesh({3, 4, 5});

		EXPECT_EQ(mesh.size(), 3*4*5);
		for(unsigned int c = 0; c < mesh.size(); c++){
			unsigned int x = c/(4*5);
			unsigned int y = (c/5)%4;
			unsigned int z = c%5;

			EXPECT_EQ(mesh[c].size(), 3);
			EXPECT_FLOAT_EQ(mesh[c][0], (x*u+y*v+z*w).x);
			EXPECT_FLOAT_EQ(mesh[c][1], (x*u+y*v+z*w).y);
			EXPECT_FLOAT_EQ(mesh[c][2], (x*u+y*v+z*w).z);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMesh.4 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMesh4){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			//Wrong argument length.
			parallelepipedCell[0].getMinorMesh({2, 3});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMeshPoint.1 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMeshPoint1){
	for(unsigned int n = 0; n < 2; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMinorMesh({3});

		for(unsigned int x = 0; x < 3; x++){
			std::vector<double> point
				= parallelepipedCell[n].getMinorMeshPoint(
					{x},
					{3}
				);

			EXPECT_EQ(point.size(), 1);
			EXPECT_FLOAT_EQ(point[0], x*2/3.);
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMeshPoint.2 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMeshPoint2){
	Vector2d u({2, 0});
	Vector2d v({-1, sqrt(3)});

	u = u/3;
	v = v/4;

	for(unsigned int n = 2; n < 4; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMinorMesh({3, 4});

		for(unsigned int x = 0; x < 3; x++){
			for(unsigned int y = 0; y < 4; y++){
				std::vector<double> point
					= parallelepipedCell[
						n
					].getMinorMeshPoint({x, y}, {3, 4});

				EXPECT_EQ(point.size(), 2);
				EXPECT_FLOAT_EQ(point[0], (x*u+y*v).x);
				EXPECT_FLOAT_EQ(point[1], (x*u+y*v).y);
			}
		}
	}
}

//TBTKFeature Uncategorized.ParallelepipedCell.getMinorMeshPoint.3 2019-11-22
TEST_F(ParallelepipedCellTest, getMinorMeshPoint3){
	Vector3d u({2, 0, 0});
	Vector3d v({-1, sqrt(3), 0});
	Vector3d w({0, 0, 2});

	u = u/3;
	v = v/4;
	w = w/5;

	for(unsigned int n = 4; n < 6; n++){
		std::vector<std::vector<double>> mesh
			= parallelepipedCell[n].getMinorMesh({3, 4, 5});

		for(unsigned int x = 0; x < 3; x++){
			for(unsigned int y = 0; y < 4; y++){
				for(unsigned int z = 0; z < 5; z++){
					std::vector<double> point
						= parallelepipedCell[
							n
						].getMinorMeshPoint(
							{x, y, z},
							{3, 4, 5}
						);

					EXPECT_EQ(point.size(), 3);
					EXPECT_FLOAT_EQ(
						point[0],
						(x*u+y*v+z*w).x
					);
					EXPECT_FLOAT_EQ(
						point[1],
						(x*u+y*v+z*w).y
					);
					EXPECT_FLOAT_EQ(
						point[2],
						(x*u+y*v+z*w).z
					);
				}
			}
		}
	}
}

};
