#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include "gtest/gtest.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

//Helper class that exposes the PropertyExtractors protected functions.
class PublicPropertyExtractor : public PropertyExtractor{
public:
	double getLowerBound() const{
		return PropertyExtractor::getLowerBound();
	}

	double getUpperBound() const{
		return PropertyExtractor::getUpperBound();
	}

	double getEnergyResolution() const{
		return PropertyExtractor::getEnergyResolution();
	}

	void calculate(
		void(*callback)(
			PropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		void *memory,
		Index pattern,
		const Index &ranges,
		int currentOffset,
		int offsetMultiplier
	){
		PropertyExtractor::calculate(
			callback,
			memory,
			pattern,
			ranges,
			currentOffset,
			offsetMultiplier
		);
	}

	template<typename DataType>
	void calculate(
		void(*callback)(
			PropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		const IndexTree &allIndices,
		const IndexTree &memoryLayout,
		Property::AbstractProperty<DataType> &abstractProperty,
		int *spinIndexHint
	){
		PropertyExtractor::calculate(
			callback,
			allIndices,
			memoryLayout,
			abstractProperty,
			spinIndexHint
		);
	}

	void setHint(void *hint){
	}

	void ensureCompliantRanges(const Index &pattern, Index &ranges){
		PropertyExtractor::ensureCompliantRanges(pattern, ranges);
	};

	void getLoopRanges(
		const Index &pattern,
		const Index &ranges,
		int *loopDimensions,
		int **loopRanges
	){
		PropertyExtractor::getLoopRanges(
			pattern,
			ranges,
			loopDimensions,
			loopRanges
		);
	}

	IndexTree generateIndexTree(
		std::initializer_list<Index> patterns,
		const HoppingAmplitudeSet &hoppingAmplitudeSet,
		bool keepSummationWildcards,
		bool keepSpinWildcards
	){
		return PropertyExtractor::generateIndexTree(
			patterns,
			hoppingAmplitudeSet,
			keepSummationWildcards,
			keepSpinWildcards
		);
	}
};

TEST(PropertyExtractor, Constructor){
	//Not testable on its own.
}

TEST(PropertyExtractor, Destructor){
	//Not testable on its own.
}

TEST(PropertyExtractor, setEnergyWindow){
	PublicPropertyExtractor propertyExtractor;

	//Verify that the energy windows is properly set.
	propertyExtractor.setEnergyWindow(-10, 10, 100);
	EXPECT_DOUBLE_EQ(propertyExtractor.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(propertyExtractor.getUpperBound(), 10);
	EXPECT_EQ(propertyExtractor.getEnergyResolution(), 100);
}

TEST(PropertyExtractor, calculateDensityRangesFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateDensity({IDX_ALL}, {0});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateDensityCustomFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateDensity({{IDX_ALL}});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateMagnetizationRangesFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateMagnetization({IDX_ALL}, {0});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateMagnetizationCustomFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateMagnetization({{IDX_ALL}});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateLDOSRangesFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateSpinPolarizedLDOS({IDX_ALL}, {0});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateSpinLDOSCustomFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateSpinPolarizedLDOS({{IDX_ALL}});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateSpinPolarizedLDOSRangesFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateSpinPolarizedLDOS({IDX_ALL}, {0});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateSpinPolarizedLDOSCustomFormat){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateSpinPolarizedLDOS({{IDX_ALL}});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateExpectationValue){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateExpectationValue({0}, {0});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateDOS){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateDOS();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, calculateEntropy){
	PropertyExtractor propertyExtractor;

	//Print error message and exit.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculateEntropy();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, getEnergyResolution){
	//Already tested through PropertyExtractor::setEnergyWindow().
}

TEST(PropertyExtractor, getLowerBound){
	//Already tested through PropertyExtractor::setEnergyWindow().
}

TEST(PropertyExtractor, getUpperBound){
	//Already tested through PropertyExtractor::setEnergyWindow().
}

//Helper function for TEST(PropertyExtractor, calculateRanges).
void callbackRanges(
	PropertyExtractor *cb_this,
	void *memory,
	const Index &index,
	int offset
){
	for(unsigned int n = 0; n < 10; n++)
		((int*)memory)[offset + n] += 10*(3*index[0] + index[2]) + n;
}

TEST(PropertyExtractor, calculateRanges){
	PublicPropertyExtractor propertyExtractor;

	//Check that the callback is called for all Indices and with the
	//correct offset when using both loop and sum specifiers as well as
	//normal subindices.
	int memory[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		memory[n] = 0;
	propertyExtractor.calculate(
		callbackRanges,
		memory,
		{IDX_SUM_ALL, 2, IDX_X},
		{2, 1, 3},
		0,
		10
	);
	for(unsigned int n = 0; n < 3*10; n++)
		EXPECT_EQ(memory[n], n + (30 + n));

	//Check that incompatible subindex specifiers generate errors.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculate(
				callbackRanges,
				memory,
				{IDX_ALL, 2, IDX_X},
				{2, 1, 3},
				0,
				10
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculate(
				callbackRanges,
				memory,
				{IDX_SUM_ALL, 2, IDX_SPIN},
				{2, 1, 3},
				0,
				10
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.calculate(
				callbackRanges,
				memory,
				{IDX_SUM_ALL, 2, IDX_SEPARATOR},
				{2, 1, 3},
				0,
				10
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//Helper function for TEST(PropertyExtractor, calculateCustom).
int spinIndex;
void callbackCustom(
	PropertyExtractor *cb_this,
	void *memory,
	const Index &index,
	int offset
){
	if(index[0] == 1 || index[0] == 3)
		EXPECT_EQ(spinIndex, 2);
	else
		EXPECT_EQ(spinIndex, 1);

	for(unsigned int n = 0; n < 100; n++){
		((SpinMatrix*)memory)[offset + n]
			+= SpinMatrix((index[0] + index[1])*n);
	}
}

TEST(PropertyExtractor, calculateCustom){
	PublicPropertyExtractor propertyExtractor;

	//Setup the Indices for which to call the callback with.
	IndexTree allIndices;
	allIndices.add({1, 0, IDX_SPIN});
	allIndices.add({1, 2, IDX_SPIN});
	allIndices.add({2, IDX_SPIN});
	allIndices.add({3, 1, IDX_SPIN});
	allIndices.add({3, 2, IDX_SPIN});
	allIndices.generateLinearMap();

	//Setup the momory layout for the property.
	IndexTree memoryLayout;
	memoryLayout.add({1, 0, IDX_SPIN});
	memoryLayout.add({1, 2, IDX_SPIN});
	memoryLayout.add({2, IDX_SPIN});
	memoryLayout.add({3, IDX_SUM_ALL, IDX_SPIN});
	memoryLayout.generateLinearMap();

	//Set the spin index. Note that since the spin subindex alternatingly
	//is in the second and third subindex position, we test whether the
	//spin index is updated properly by the calculate function.
	propertyExtractor.setHint(&spinIndex);

	//Create the property.
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		-10,
		10,
		100
	);

	//Run calculation.
	propertyExtractor.calculate(
		callbackCustom,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		&spinIndex
	);

	//Check the results.
	for(unsigned int n = 0; n < 100; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				//Real part.
				EXPECT_NEAR(
					real(spinPolarizedLDOS(0 + n).at(r, c)),
					(1 + 0)*n,
					EPSILON_100
				);
				EXPECT_NEAR(
					real(spinPolarizedLDOS(100 + n).at(r, c)),
					(1 + 2)*n,
					EPSILON_100
				);
				EXPECT_NEAR(
					real(spinPolarizedLDOS(200 + n).at(r, c)),
					(2 + IDX_SPIN)*n,
					EPSILON_100
				);
				EXPECT_NEAR(
					real(spinPolarizedLDOS(300 + n). at(r, c)),
					((3 + 1) + (3 + 2))*n,
					EPSILON_100
				);

				//Imaginary part.
				EXPECT_NEAR(
					imag(spinPolarizedLDOS(0 + n).at(r, c)),
					0,
					EPSILON_100
				);
				EXPECT_NEAR(
					imag(spinPolarizedLDOS(100 + n).at(r, c)),
					0,
					EPSILON_100
				);
				EXPECT_NEAR(
					imag(spinPolarizedLDOS(200 + n).at(r, c)),
					0,
					EPSILON_100
				);
				EXPECT_NEAR(
					imag(spinPolarizedLDOS(300 + n). at(r, c)),
					0,
					EPSILON_100
				);
			}
		}
	}

	//Clear the spin index.
	propertyExtractor.setHint(nullptr);
}

TEST(PropertyExtractor, enureCompliantRanges){
	PublicPropertyExtractor propertyExtractor;

	//Check that the range for each subindex that is not a specifier in the
	//pattern Index is set to one.
	Index pattern({3, IDX_ALL, 0, IDX_SPIN, 2, IDX_X, IDX_Y, IDX_Z});
	Index ranges({10, 10, 10, 10, 10, 10, 10, 10});
	propertyExtractor.ensureCompliantRanges(pattern, ranges);
	EXPECT_EQ(ranges[0], 1);
	EXPECT_EQ(ranges[1], 10);
	EXPECT_EQ(ranges[2], 1);
	EXPECT_EQ(ranges[3], 10);
	EXPECT_EQ(ranges[4], 1);
	EXPECT_EQ(ranges[5], 10);
	EXPECT_EQ(ranges[6], 10);
	EXPECT_EQ(ranges[7], 10);
}

TEST(PropertyExtractor, getLoopRanges){
	PublicPropertyExtractor propertyExtractor;

	//Check that IDX_X, IDX_Y, and IDX_Z are identified as three loop
	//subindices and that their ranges are extracted.
	Index pattern({3, IDX_ALL, 0, IDX_SPIN, 2, IDX_X, IDX_Y, IDX_Z});
	Index ranges({2, 3, 4, 5, 6, 7, 8, 9});
	int loopDimensions;
	int *loopRanges;
	propertyExtractor.getLoopRanges(
		pattern,
		ranges,
		&loopDimensions,
		&loopRanges
	);
	EXPECT_EQ(loopDimensions, 3);
	EXPECT_EQ(loopRanges[0], 7);
	EXPECT_EQ(loopRanges[1], 8);
	EXPECT_EQ(loopRanges[2], 9);

	delete [] loopRanges;
}

TEST(PropertyExtractor, gnerateIndexTree){
	PublicPropertyExtractor propertyExtractor;

	HoppingAmplitudeSet hoppingAmplitudeSet;
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {0, 0, 0}, {0, 0, 0}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {0, 0, 1}, {0, 0, 1}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {0, 0, 2}, {0, 0, 2}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {1, 0, 0}, {1, 0, 0}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {1, 1, 0}, {1, 1, 0}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {1, 0, 1}, {1, 0, 1}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {1, 1, 1}, {1, 1, 1}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {2, 0}, {2, 0}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {2, 1}, {2, 1}));
	hoppingAmplitudeSet.add(HoppingAmplitude(0, {2, 1}, {2, 2}));

	//Check that generation without preservation of any wildcards works.
	IndexTree indexTree0 = propertyExtractor.generateIndexTree(
		{
			{0, 0, IDX_ALL},
			{1, IDX_SPIN, 1},
			{2, IDX_SUM_ALL}
		},
		hoppingAmplitudeSet,
		false,
		false
	);
	IndexTree::ConstIterator iterator0 = indexTree0.cbegin();
	EXPECT_TRUE((*iterator0).equals({0, 0, 0}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({0, 0, 1}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({0, 0, 2}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({1, 0, 1}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({1, 1, 1}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({2, 0}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({2, 1}));
	++iterator0;
	EXPECT_TRUE((*iterator0).equals({2, 2}));
	++iterator0;
	EXPECT_TRUE(iterator0 == indexTree0.cend());

	//Check that generation while preserving summation indices works.
	IndexTree indexTree1 = propertyExtractor.generateIndexTree(
		{
			{0, 0, IDX_ALL},
			{1, IDX_SPIN, 1},
			{2, IDX_SUM_ALL}
		},
		hoppingAmplitudeSet,
		true,
		false
	);
	IndexTree::ConstIterator iterator1 = indexTree1.cbegin();
	EXPECT_TRUE((*iterator1).equals({0, 0, 0}));
	++iterator1;
	EXPECT_TRUE((*iterator1).equals({0, 0, 1}));
	++iterator1;
	EXPECT_TRUE((*iterator1).equals({0, 0, 2}));
	++iterator1;
	EXPECT_TRUE((*iterator1).equals({1, 0, 1}));
	++iterator1;
	EXPECT_TRUE((*iterator1).equals({1, 1, 1}));
	++iterator1;
	EXPECT_TRUE((*iterator1).equals({2, IDX_SUM_ALL}));
	++iterator1;
	EXPECT_TRUE(iterator1 == indexTree1.cend());

	//Check that generation while preserving spin wildcards works.
	IndexTree indexTree2 = propertyExtractor.generateIndexTree(
		{
			{0, 0, IDX_ALL},
			{1, IDX_SPIN, 1},
			{2, IDX_SUM_ALL}
		},
		hoppingAmplitudeSet,
		false,
		true
	);
	IndexTree::ConstIterator iterator2 = indexTree2.cbegin();
	EXPECT_TRUE((*iterator2).equals({0, 0, 0}));
	++iterator2;
	EXPECT_TRUE((*iterator2).equals({0, 0, 1}));
	++iterator2;
	EXPECT_TRUE((*iterator2).equals({0, 0, 2}));
	++iterator2;
	EXPECT_TRUE((*iterator2).equals({1, IDX_SPIN, 1}));
	++iterator2;
	EXPECT_TRUE((*iterator2).equals({2, 0}));
	++iterator2;
	EXPECT_TRUE((*iterator2).equals({2, 1}));
	++iterator2;
	EXPECT_TRUE((*iterator2).equals({2, 2}));
	++iterator2;
	EXPECT_TRUE(iterator2 == indexTree2.cend());

	//Check that generation while preserving both summation and spin
	//indices works.
	IndexTree indexTree3 = propertyExtractor.generateIndexTree(
		{
			{0, 0, IDX_ALL},
			{1, IDX_SPIN, 1},
			{2, IDX_SUM_ALL}
		},
		hoppingAmplitudeSet,
		true,
		true
	);
	IndexTree::ConstIterator iterator3 = indexTree3.cbegin();
	EXPECT_TRUE((*iterator3).equals({0, 0, 0}));
	++iterator3;
	EXPECT_TRUE((*iterator3).equals({0, 0, 1}));
	++iterator3;
	EXPECT_TRUE((*iterator3).equals({0, 0, 2}));
	++iterator3;
	EXPECT_TRUE((*iterator3).equals({1, IDX_SPIN, 1}));
	++iterator3;
	EXPECT_TRUE((*iterator3).equals({2, IDX_SUM_ALL}));
	++iterator3;
	EXPECT_TRUE(iterator3 == indexTree3.cend());

}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
