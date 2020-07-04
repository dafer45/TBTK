#include "TBTK/Functions.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"
#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Solver/Diagonalizer.h"

#include "gtest/gtest.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

//Helper class that exposes the PropertyExtractors protected functions.
class PublicPropertyExtractor : public PropertyExtractor{
public:
	class PublicInformation : public Information{
	public:
		void setSpinIndex(int spinIndex){
			Information::setSpinIndex(spinIndex);
		}

		int getSpinIndex() const{
			return Information::getSpinIndex();
		}
	};

	const Range& getEnergyWindow() const{
		return PropertyExtractor::getEnergyWindow();
	}

	int getLowerFermionicMatsubaraEnergyIndex() const{
		return PropertyExtractor::getLowerFermionicMatsubaraEnergyIndex();
	}

	int getUpperFermionicMatsubaraEnergyIndex() const{
		return PropertyExtractor::getUpperFermionicMatsubaraEnergyIndex();
	}

	int getLowerBosonicMatsubaraEnergyIndex() const{
		return PropertyExtractor::getLowerBosonicMatsubaraEnergyIndex();
	}

	int getUpperBosonicMatsubaraEnergyIndex() const{
		return PropertyExtractor::getUpperBosonicMatsubaraEnergyIndex();
	}

	template<typename DataType>
	void calculate(
		void(*callback)(
			PropertyExtractor *cb_this,
			Property::Property &property,
			const Index &index,
			int offset,
			Information &information
		),
		Property::AbstractProperty<DataType> &property,
		Index pattern,
		const Index &ranges,
		int currentOffset,
		int offsetMultiplier,
		Information &information
	){
		PropertyExtractor::calculate(
			callback,
			property,
			pattern,
			ranges,
			currentOffset,
			offsetMultiplier,
			information
		);
	}

	template<typename DataType>
	void calculate(
		void(*callback)(
			PropertyExtractor *cb_this,
			Property::Property &property,
			const Index &index,
			int offset,
			Information &information
		),
		const IndexTree &allIndices,
		const IndexTree &memoryLayout,
		Property::AbstractProperty<DataType> &abstractProperty,
		Information &information
	){
		PropertyExtractor::calculate(
			callback,
			allIndices,
			memoryLayout,
			abstractProperty,
			information
		);
	}

	void ensureCompliantRanges(const Index &pattern, Index &ranges){
		PropertyExtractor::ensureCompliantRanges(pattern, ranges);
	};

	std::vector<int> getLoopRanges(
		const Index &pattern,
		const Index &ranges
	){
		return PropertyExtractor::getLoopRanges(
			pattern,
			ranges
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

	void setSolver(Solver::Solver &solver){
		PropertyExtractor::setSolver(solver);
	}

	template<typename SolverType>
	SolverType& getSolver(){
		return PropertyExtractor::getSolver<SolverType>();
	}

	static double getThermodynamicEquilibriumOccupation(
		double energy,
		const Model &model
	){
		return PropertyExtractor::getThermodynamicEquilibriumOccupation(
			energy,
			model
		);
	}

	//Helper function for TEST(PropertyExtractor, calculateRanges).
	static void callbackRanges(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	){
		Property::LDOS &ldos = (Property::LDOS&)property;
		std::vector<double> &data = ldos.getDataRW();

		for(unsigned int n = 0; n < 10; n++)
			data[offset + n] += 10*(3*index[0] + index[2]) + n;
	}

	//Helper function for TEST(PropertyExtractor, calculateCustom).
	static void callbackCustom(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	){
		Property::SpinPolarizedLDOS &spinPolarizedLDOS
			= (Property::SpinPolarizedLDOS&)property;
		std::vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();

		if(index[0] == 1 || index[0] == 3)
			EXPECT_EQ(information.getSpinIndex(), 2);
		else
			EXPECT_EQ(information.getSpinIndex(), 1);

		for(unsigned int n = 0; n < 100; n++){
			data[offset + n]
				+= SpinMatrix((index[0] + index[1])*n);
		}
	}
};

TEST(PropertyExtractor, Constructor){
	//Not testable on its own.
}

TEST(PropertyExtractor, Destructor){
	//Not testable on its own.
}

TEST(PropertyExtractor, setEnergyWindowReal0){
	PublicPropertyExtractor propertyExtractor;

	//Verify that the energy windows is properly set.
	propertyExtractor.setEnergyWindow(-10, 10, 100);
	const Range &energyWindow = propertyExtractor.getEnergyWindow();
	EXPECT_DOUBLE_EQ(energyWindow[0], -10);
	EXPECT_DOUBLE_EQ(energyWindow.getLast(), 10);
	EXPECT_EQ(energyWindow.getResolution(), 100);

	//Print error message when accessing Matsubara quantities.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getLowerFermionicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getUpperFermionicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getLowerBosonicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getUpperBosonicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, setEnergyWindowReal1){
	PublicPropertyExtractor propertyExtractor;

	//Verify that the energy windows is properly set.
	propertyExtractor.setEnergyWindow(Range(-10, 10, 100));
	const Range &energyWindow = propertyExtractor.getEnergyWindow();
	EXPECT_DOUBLE_EQ(energyWindow[0], -10);
	EXPECT_DOUBLE_EQ(energyWindow.getLast(), 10);
	EXPECT_EQ(energyWindow.getResolution(), 100);

	//Print error message when accessing Matsubara quantities.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getLowerFermionicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getUpperFermionicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getLowerBosonicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getUpperBosonicMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(PropertyExtractor, setEnergyWindowMatsubara){
	PublicPropertyExtractor propertyExtractor;

	//Verify that the energy windows is properly set.
	propertyExtractor.setEnergyWindow(-11, 11, -10, 10);
	EXPECT_DOUBLE_EQ(
		propertyExtractor.getLowerFermionicMatsubaraEnergyIndex(),
		-11
	);
	EXPECT_DOUBLE_EQ(
		propertyExtractor.getUpperFermionicMatsubaraEnergyIndex(),
		11
	);
	EXPECT_DOUBLE_EQ(
		propertyExtractor.getLowerBosonicMatsubaraEnergyIndex(),
		-10
	);
	EXPECT_DOUBLE_EQ(
		propertyExtractor.getUpperBosonicMatsubaraEnergyIndex(),
		10
	);

	//Print error message for even Fermionic indices, and odd Bosonic
	//indices.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.setEnergyWindow(-10, 11, -10, 10);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.setEnergyWindow(-11, 10, -10, 10);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.setEnergyWindow(-11, 11, -11, 10);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.setEnergyWindow(-11, 11, -10, 11);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Print error message if the lower index is larger than the upper
	//index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.setEnergyWindow(11, -11, -10, 10);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.setEnergyWindow(-11, 11, 10, -10);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Print error message when accessing real quantities.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getEnergyWindow();
		},
		::testing::ExitedWithCode(1),
		""
	);
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

TEST(PropertyExtractor, getEnergyType){
	//Not testable on its own.
}

TEST(PropertyExtractor, getEnergyResolution){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, getLowerBound){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, getUpperBound){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, getLowerFermionicMatsubaraEnergyIndex){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, getUpperFermionicMatsubaraEnergyIndex){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, getLowerBosonicMatsubaraEnergyIndex){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, getUpperBosonicMatsubaraEnergyIndex){
	//Already tested through
	//PropertyExtractor::setEnergyWindowReal.
	//PropertyExtractor::setEnergyWindowMatsubara.
}

TEST(PropertyExtractor, calculateRanges){
	PublicPropertyExtractor propertyExtractor;

	//Check that the callback is called for all Indices and with the
	//correct offset when using both loop and sum specifiers as well as
	//normal subindices.
	Property::LDOS ldos({2, 1, 3}, Range(-1, 1, 10));
	for(unsigned int n = 0; n < ldos.getSize(); n++)
		ldos.getDataRW()[n] = 0;
	PublicPropertyExtractor::PublicInformation information;
	propertyExtractor.calculate(
		PublicPropertyExtractor::callbackRanges,
		ldos,
		{IDX_SUM_ALL, 2, IDX_X},
		{2, 1, 3},
		0,
		10,
		information
	);
	for(unsigned int n = 0; n < 3*10; n++)
		EXPECT_NEAR(ldos.getData()[n], n + (30 + n), EPSILON_100);

	//Check that incompatible subindex specifiers generate errors.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PublicPropertyExtractor::PublicInformation information;
			propertyExtractor.calculate(
				PublicPropertyExtractor::callbackRanges,
				ldos,
				{IDX_ALL, 2, IDX_X},
				{2, 1, 3},
				0,
				10,
				information
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PublicPropertyExtractor::PublicInformation information;
			propertyExtractor.calculate(
				PublicPropertyExtractor::callbackRanges,
				ldos,
				{IDX_SUM_ALL, 2, IDX_SPIN},
				{2, 1, 3},
				0,
				10,
				information
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PublicPropertyExtractor::PublicInformation information;
			propertyExtractor.calculate(
				PublicPropertyExtractor::callbackRanges,
				ldos,
				{IDX_SUM_ALL, 2, IDX_SEPARATOR},
				{2, 1, 3},
				0,
				10,
				information
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
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

	//Create the property.
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		Range(-10, 10, 100)
	);

	//Run calculation.
	PublicPropertyExtractor::PublicInformation information;
	propertyExtractor.calculate(
		PublicPropertyExtractor::callbackCustom,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		information
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
	std::vector<int> loopRanges = propertyExtractor.getLoopRanges(
		pattern,
		ranges
	);
	EXPECT_EQ(loopRanges.size(), 3);
	EXPECT_EQ(loopRanges[0], 7);
	EXPECT_EQ(loopRanges[1], 8);
	EXPECT_EQ(loopRanges[2], 9);
}

TEST(PropertyExtractor, gnerateIndexTree){
	PublicPropertyExtractor propertyExtractor;

	//Setup a HoppingAmplitudeSet to extract Indices from.
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

	//Check that two compound indices works.
	IndexTree indexTree4 = propertyExtractor.generateIndexTree(
		{{{0, 0, IDX_ALL}, {0, 0, IDX_ALL}}},
		hoppingAmplitudeSet,
		false,
		false
	);
	IndexTree::ConstIterator iterator4 = indexTree4.cbegin();
	EXPECT_TRUE((*iterator4).equals({{0, 0, 0}, {0, 0, 0}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 0}, {0, 0, 1}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 0}, {0, 0, 2}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 1}, {0, 0, 0}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 1}, {0, 0, 1}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 1}, {0, 0, 2}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 2}, {0, 0, 0}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 2}, {0, 0, 1}}));
	++iterator4;
	EXPECT_TRUE((*iterator4).equals({{0, 0, 2}, {0, 0, 2}}));
	++iterator4;
	EXPECT_TRUE(iterator4 == indexTree4.cend());
}

//TBTKFeature PropertyExtractor.PropertyExtractor.setSolver.0 2020-06-07
TEST(PropertyExtractor, setSolver0){
	//Tested through PropertyExtractor::getSolver
}

//TBTKFeature PropertyExtractor.PropertyExtractor.getSolver.0 2020-06-07
TEST(PropertyExtractor, getSolver0){
	PublicPropertyExtractor propertyExtractor;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getSolver<Solver::Solver>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PropertyExtractor.getSolver.1 2020-06-07
TEST(PropertyExtractor, getSolver1){
	Solver::Diagonalizer solver;
	PublicPropertyExtractor propertyExtractor;
	propertyExtractor.setSolver(solver);
	Solver::Solver &retrievedSolver
		= propertyExtractor.getSolver<Solver::Solver>();

	EXPECT_EQ(
		&dynamic_cast<Solver::Diagonalizer&>(retrievedSolver),
		&solver
	);
}

//TBTKFeature PropertyExtractor.PropertyExtractor.getSolver.2 2020-06-07
TEST(PropertyExtractor, getSolver2){
	Solver::Diagonalizer solver;
	PublicPropertyExtractor propertyExtractor;
	propertyExtractor.setSolver(solver);
	Solver::Diagonalizer &retrievedSolver
		= propertyExtractor.getSolver<Solver::Diagonalizer>();

	EXPECT_EQ(&retrievedSolver, &solver);
}

//TBTKFeature PropertyExtractor.PropertyExtractor.getSolver.3 2020-06-07
TEST(PropertyExtractor, getSolver3){
	Solver::Diagonalizer solver;
	PublicPropertyExtractor propertyExtractor;
	propertyExtractor.setSolver(solver);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			propertyExtractor.getSolver<Solver::ArnoldiIterator>();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PropertyExtractor.getThermodynamicEquilibriumOccupation.0 2020-07-04
TEST(PropertyExtractor, getThermodynamicEquilibirumOccupation0){
	const double CHEMICAL_POTENTIAL = 7;
	const double TEMPERATURE = 18;
	Model model;
	model.setStatistics(Statistics::FermiDirac);
	model.setChemicalPotential(CHEMICAL_POTENTIAL);
	model.setTemperature(TEMPERATURE);

	for(int n = 0; n < 10; n++){
		double energy = CHEMICAL_POTENTIAL + (n-5)/1000.;
		EXPECT_NEAR(
			PublicPropertyExtractor::getThermodynamicEquilibriumOccupation(
				energy,
				model
			),
			Functions::fermiDiracDistribution(
				energy,
				CHEMICAL_POTENTIAL,
				TEMPERATURE
			),
			EPSILON_100
		);
	}
}

//TBTKFeature PropertyExtractor.PropertyExtractor.getThermodynamicEquilibriumOccupation.1 2020-07-04
TEST(PropertyExtractor, getThermodynamicEquilibirumOccupation1){
	const double CHEMICAL_POTENTIAL = 7;
	const double TEMPERATURE = 18;
	Model model;
	model.setStatistics(Statistics::BoseEinstein);
	model.setChemicalPotential(CHEMICAL_POTENTIAL);
	model.setTemperature(TEMPERATURE);

	for(int n = 0; n < 10; n++){
		double energy = CHEMICAL_POTENTIAL + (n+1)/1000.;
		EXPECT_NEAR(
			PublicPropertyExtractor::getThermodynamicEquilibriumOccupation(
				energy,
				model
			),
			Functions::boseEinsteinDistribution(
				energy,
				CHEMICAL_POTENTIAL,
				TEMPERATURE
			),
			EPSILON_100
		);
	}
}

TEST(PropertyExtractor, Information){
	//Nothing to test.
}

TEST(PropertyExtractorInformation, setSpinIndex){
	//Tested through PropertyExtractorInformation::getSpinIndex.
}

TEST(PropertyExtractorInformation, getSpinIndex){
	PublicPropertyExtractor::PublicInformation information;

	//Check that the value is initialized to -1.
	EXPECT_EQ(information.getSpinIndex(), -1);

	//Check that the value can be set and retreived.
	information.setSpinIndex(3);
	EXPECT_EQ(information.getSpinIndex(), 3);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
