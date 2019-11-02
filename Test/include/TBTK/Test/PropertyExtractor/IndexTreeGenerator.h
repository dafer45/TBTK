#include "TBTK/PropertyExtractor/IndexTreeGenerator.h"

#include "gtest/gtest.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

class IndexTreeGeneratorTest : public ::testing::Test{
protected:
	Model model;

	void SetUp() override{
		for(unsigned int x = 0; x < 2; x++){
			for(unsigned int y = 0; y < 2; y++){
				for(unsigned int spin = 0; spin < 2; spin++){
					model << HoppingAmplitude(
						0,
						{x, y, spin},
						{x, y, spin}
					);
				}
			}
		}
		model.construct();
	}
};

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.1 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate1){
	IndexTreeGenerator indexTreeGenerator(model);
	IndexTree indexTree = indexTreeGenerator.generate({
		{1, IDX_ALL, IDX_SUM_ALL}
	});
	EXPECT_EQ(indexTree.getSize(), 4);
	for(unsigned int y = 0; y < 2; y++)
		for(unsigned int spin = 0; spin < 2; spin++)
			EXPECT_TRUE(indexTree.contains({1, y, spin}));
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.2 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate2){
	IndexTreeGenerator indexTreeGenerator(model);
	indexTreeGenerator.setKeepSummationWildcards(true);
	IndexTree indexTree = indexTreeGenerator.generate({
		{1, IDX_ALL, IDX_SUM_ALL}
	});
	EXPECT_EQ(indexTree.getSize(), 2);
	for(unsigned int y = 0; y < 2; y++)
		EXPECT_TRUE(indexTree.contains({1, y, IDX_SUM_ALL}));
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.3 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate3){
	IndexTreeGenerator indexTreeGenerator(model);
	IndexTree indexTree = indexTreeGenerator.generate({
		{1, IDX_ALL, IDX_SPIN}
	});
	EXPECT_EQ(indexTree.getSize(), 4);
	for(unsigned int y = 0; y < 2; y++)
		for(unsigned int spin = 0; spin < 2; spin++)
			EXPECT_TRUE(indexTree.contains({1, y, spin}));
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.4 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate4){
	IndexTreeGenerator indexTreeGenerator(model);
	indexTreeGenerator.setKeepSpinWildcards(true);
	IndexTree indexTree = indexTreeGenerator.generate({
		{1, IDX_ALL, IDX_SPIN}
	});
	EXPECT_EQ(indexTree.getSize(), 2);
	for(unsigned int y = 0; y < 2; y++)
		EXPECT_TRUE(indexTree.contains({1, y, IDX_SPIN}));
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.5 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate5){
	IndexTreeGenerator indexTreeGenerator(model);
	IndexTree indexTree = indexTreeGenerator.generate({
		{{1, IDX_ALL, IDX_SUM_ALL}, {1, IDX_ALL, IDX_SUM_ALL}}
	});
	EXPECT_EQ(indexTree.getSize(), 16);
	for(unsigned int y0 = 0; y0 < 2; y0++){
		for(unsigned int spin0 = 0; spin0 < 2; spin0++){
			for(unsigned int y1 = 0; y1 < 2; y1++){
				for(
					unsigned int spin1 = 0;
					spin1 < 2;
					spin1++
				){
					EXPECT_TRUE(
						indexTree.contains({
							{1, y0, spin0},
							{1, y1, spin1}
						})
					);
				}
			}
		}
	}
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.6 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate6){
	IndexTreeGenerator indexTreeGenerator(model);
	indexTreeGenerator.setKeepSummationWildcards(true);
	IndexTree indexTree = indexTreeGenerator.generate({
		{{1, IDX_ALL, IDX_SUM_ALL}, {1, IDX_ALL, IDX_SUM_ALL}}
	});
	EXPECT_EQ(indexTree.getSize(), 4);
	for(unsigned int y0 = 0; y0 < 2; y0++){
		for(unsigned int y1 = 0; y1 < 2; y1++){
			EXPECT_TRUE(
				indexTree.contains({
					{1, y0, IDX_SUM_ALL},
					{1, y1, IDX_SUM_ALL}
				})
			);
		}
	}
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.7 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate7){
	IndexTreeGenerator indexTreeGenerator(model);
	IndexTree indexTree = indexTreeGenerator.generate({
		{{1, IDX_ALL, IDX_SPIN}, {1, IDX_ALL, IDX_SPIN}}
	});
	EXPECT_EQ(indexTree.getSize(), 16);
	for(unsigned int y0 = 0; y0 < 2; y0++){
		for(unsigned int spin0 = 0; spin0 < 2; spin0++){
			for(unsigned int y1 = 0; y1 < 2; y1++){
				for(
					unsigned int spin1 = 0;
					spin1 < 2;
					spin1++
				){
					EXPECT_TRUE(
						indexTree.contains({
							{1, y0, spin0},
							{1, y1, spin1}
						})
					);
				}
			}
		}
	}
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.8 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate8){
	IndexTreeGenerator indexTreeGenerator(model);
	indexTreeGenerator.setKeepSpinWildcards(true);
	IndexTree indexTree = indexTreeGenerator.generate({
		{{1, IDX_ALL, IDX_SPIN}, {1, IDX_ALL, IDX_SPIN}}
	});
	EXPECT_EQ(indexTree.getSize(), 4);
	for(unsigned int y0 = 0; y0 < 2; y0++){
		for(unsigned int y1 = 0; y1 < 2; y1++){
			EXPECT_TRUE(
				indexTree.contains(
					{{1, y0, IDX_SPIN}, {1, y1, IDX_SPIN}}
				)
			);
		}
	}
}

//TBTKFeature PropertyExtractor.IndexTreeGenerator.generate.9 2019-11-02
TEST_F(IndexTreeGeneratorTest, generate9){
	IndexTreeGenerator indexTreeGenerator(model);
	indexTreeGenerator.setKeepSpinWildcards(true);
	IndexTree indexTree = indexTreeGenerator.generate({
		{
			{1, IDX_ALL, IDX_SPIN},
			{1, IDX_ALL, IDX_SPIN},
			{1, IDX_ALL, IDX_SPIN}
		}
	});
	EXPECT_EQ(indexTree.getSize(), 8);
	for(unsigned int y0 = 0; y0 < 2; y0++){
		for(unsigned int y1 = 0; y1 < 2; y1++){
			EXPECT_TRUE(
				indexTree.contains(
					{
						{1, y0, IDX_SPIN},
						{1, y1, IDX_SPIN},
						{1, y1, IDX_SPIN}
					}
				)
			);
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
