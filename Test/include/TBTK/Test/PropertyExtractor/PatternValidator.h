#include "TBTK/PropertyExtractor/PatternValidator.h"

#include "gtest/gtest.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

//TBTKFeature PropertyExtractor.PatternValidator.chechNumRequiredComponentIndices.1 2019-11-01
TEST(PatternValidator, checkNumRequiredComponentIndices1){
	PatternValidator patternValidator;
	patternValidator.validate({
		{1, 2},
		{{1, 2}, {1, 2}}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechNumRequiredComponentIndices.2 2019-11-01
TEST(PatternValidator, checkNumRequiredComponentIndices2){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(2);
	patternValidator.validate({
		{{1, 2}, {1, 2}}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechNumRequiredComponentIndices.3 2019-11-01
TEST(PatternValidator, checkNumRequiredComponentIndices3){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(2);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{1, 2}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.1 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags1){
	PatternValidator patternValidator;
	patternValidator.validate({
		{1, 2}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.2 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags2){
	PatternValidator patternValidator;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{1, IDX_ALL}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.3 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags3){
	PatternValidator patternValidator;
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
	patternValidator.validate({
		{1, 2}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.4 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags4){
	PatternValidator patternValidator;
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
	patternValidator.validate({
		{IDX_ALL, IDX_SPIN}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.5 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags5){
	PatternValidator patternValidator;
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{IDX_ALL, IDX_SUM_ALL}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
